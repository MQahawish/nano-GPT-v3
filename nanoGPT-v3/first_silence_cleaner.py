import os
import re
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import time


def clean_initial_silence(content: str) -> str:
    """
    Remove the first 'sepxx d_' event after 'START' in the MIDI encoding.
    
    Args:
        content: The MIDI encoding content
        
    Returns:
        Cleaned MIDI encoding with initial silence removed
    """
    # Pattern to match: START followed by sepxx and then a d_ event
    pattern = r"(START\s+sepxx\s+d\d+\s+)"
    
    # Replace the first occurrence only
    cleaned_content = re.sub(pattern, "START ", content, count=1)
    
    return cleaned_content


def process_file(file_path: str, dry_run: bool = False) -> Tuple[str, bool, str]:
    """
    Process a single file to remove initial silence.
    
    Args:
        file_path: Path to the file
        dry_run: If True, don't actually modify the file
        
    Returns:
        Tuple of (file_path, was_modified, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file starts with START and has an initial silence
        if not re.search(r"START\s+sepxx\s+d\d+\s+", content):
            return file_path, False, "no initial silence pattern"
        
        cleaned_content = clean_initial_silence(content)
        
        # Check if content was actually modified
        if cleaned_content == content:
            return file_path, False, "no changes needed"
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            return file_path, True, "cleaned"
        else:
            return file_path, True, "would clean (dry run)"
    
    except Exception as e:
        return file_path, False, f"error: {str(e)}"


def process_file_batch(batch: List[str], dry_run: bool) -> List[Tuple[str, bool, str]]:
    """
    Process a batch of files.
    
    Args:
        batch: List of file paths
        dry_run: If True, don't modify files
    
    Returns:
        List of results, each being a tuple of (file_path, was_modified, message)
    """
    results = []
    for file_path in batch:
        result = process_file(file_path, dry_run)
        results.append(result)
    return results


def get_all_txt_files(directory_paths: List[str], recursive: bool = True) -> List[str]:
    """
    Get all .txt files from the given directories.
    
    Args:
        directory_paths: List of directory paths to search
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    all_files = []
    
    for directory_path in directory_paths:
        if not os.path.exists(directory_path):
            print(f"Warning: Directory '{directory_path}' does not exist, skipping")
            continue
        
        if recursive:
            # Walk through all subdirectories
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
        else:
            # Only search top level directory
            for file in os.listdir(directory_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path):
                        all_files.append(file_path)
    
    return all_files


def main():
    parser = argparse.ArgumentParser(description='Clean initial silence from MIDI encodings in parallel')
    parser.add_argument('--dirs', nargs='+', required=True, help='List of directories containing encoded MIDI files')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive directory scanning')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without modifying files')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=500, 
                       help='Number of files to process in each batch (default: 100)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Determine number of workers
    workers = args.workers if args.workers else multiprocessing.cpu_count()
    print(f"Using {workers} worker processes with batch size {args.batch_size}")
    
    # Get all txt files from the directories
    print("Finding files...")
    all_files = get_all_txt_files(args.dirs, recursive=not args.no_recursive)
    print(f"Found {len(all_files)} files to process")
    
    # Prepare batches
    batches = [all_files[i:i + args.batch_size] for i in range(0, len(all_files), args.batch_size)]
    print(f"Split into {len(batches)} batches")
    
    # Process batches in parallel
    modified_count = 0
    processed_count = 0
    error_count = 0
    
    print("Processing files in parallel...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches to be processed
        future_to_batch = {
            executor.submit(process_file_batch, batch, args.dry_run): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_batch)):
            batch_idx = future_to_batch[future]
            try:
                results = future.result()
                
                # Count results
                batch_modified = sum(1 for _, modified, _ in results if modified)
                batch_errors = sum(1 for _, _, msg in results if msg.startswith("error"))
                
                modified_count += batch_modified
                error_count += batch_errors
                processed_count += len(results)
                
                # Print progress every 10 batches or for the last batch
                if (i + 1) % 10 == 0 or i == len(batches) - 1:
                    elapsed = time.time() - start_time
                    percent = processed_count / len(all_files) * 100
                    files_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    
                    print(f"Progress: {processed_count}/{len(all_files)} files ({percent:.1f}%) - "
                          f"{modified_count} modified, {error_count} errors - "
                          f"{files_per_sec:.1f} files/sec")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                error_count += len(batches[batch_idx])
                processed_count += len(batches[batch_idx])
    
    # Final report
    elapsed = time.time() - start_time
    print(f"\nFinished processing in {elapsed:.1f} seconds.")
    print(f"Total files: {len(all_files)}")
    print(f"Modified: {modified_count}")
    print(f"Errors: {error_count}")
    print(f"Average speed: {len(all_files) / elapsed:.1f} files/sec")


if __name__ == "__main__":
    main()