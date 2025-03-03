import os
import json
import re
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Dict, List, Set, Union, Tuple


def generate_n_v_tokens():
    """
    Generate all n_ and v_ tokens in the range 0-127.
    
    Returns:
        Set of all n and v tokens
    """
    tokens = set()
    
    # Add n0 to n127
    for i in range(128):
        tokens.add(f"n{i}")
    
    # Add v0 to v127
    for i in range(128):
        tokens.add(f"v{i}")
    
    return tokens


def extract_d_tokens_from_file(file_path: str) -> Set[str]:
    """
    Extract all d_ tokens from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Set of d_ tokens found in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all d tokens (d followed by numbers)
        d_tokens = re.findall(r'd\d+', content)
        
        return set(d_tokens)
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return set()


def process_file_batch(batch: List[str]) -> Set[str]:
    """
    Process a batch of files to extract d tokens.
    
    Args:
        batch: List of file paths
    
    Returns:
        Set of d tokens from all files in the batch
    """
    all_d_tokens = set()
    for file_path in batch:
        file_tokens = extract_d_tokens_from_file(file_path)
        all_d_tokens.update(file_tokens)
    return all_d_tokens


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


def build_tokenizer_parallel(directory_paths: List[str], recursive: bool = True, 
                            workers: int = None, batch_size: int = 100) -> Dict:
    """
    Build a tokenizer for MIDI encodings using parallel processing.
    
    Args:
        directory_paths: List of directory paths to search
        recursive: Whether to search recursively
        workers: Number of worker processes
        batch_size: Number of files to process in each batch
        
    Returns:
        Tokenizer dictionary with itos and stoi mappings
    """
    start_time = time.time()
    
    # Start with known tokens
    all_tokens = set(["START", "sepxx"])
    
    # Add n and v tokens
    all_tokens.update(generate_n_v_tokens())
    
    # Get all txt files
    print("Finding files...")
    all_files = get_all_txt_files(directory_paths, recursive)
    print(f"Found {len(all_files)} files to process")
    
    # Prepare batches
    batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    print(f"Split into {len(batches)} batches")
    
    # Determine number of workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    print(f"Using {workers} worker processes")
    
    # Process batches in parallel
    all_d_tokens = set()
    processed_count = 0
    
    print("Processing files in parallel...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches to be processed
        future_to_batch = {
            executor.submit(process_file_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_batch)):
            batch_idx = future_to_batch[future]
            try:
                batch_tokens = future.result()
                all_d_tokens.update(batch_tokens)
                
                processed_count += len(batches[batch_idx])
                
                # Print progress every 10 batches or for the last batch
                if (i + 1) % 10 == 0 or i == len(batches) - 1:
                    elapsed = time.time() - start_time
                    percent = processed_count / len(all_files) * 100
                    files_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    
                    print(f"Progress: {processed_count}/{len(all_files)} files ({percent:.1f}%) - "
                          f"Found {len(all_d_tokens)} unique d tokens - "
                          f"{files_per_sec:.1f} files/sec")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                processed_count += len(batches[batch_idx])
    
    # Add d tokens to the vocabulary
    all_tokens.update(all_d_tokens)
    
    # Create sorted list for deterministic ordering
    sorted_tokens = sorted(list(all_tokens))
    
    # Create mappings
    itos = {i: token for i, token in enumerate(sorted_tokens)}
    stoi = {token: i for i, token in enumerate(sorted_tokens)}
    
    # Create tokenizer dictionary
    tokenizer = {
        "vocab_size": len(sorted_tokens),
        "itos": itos,
        "stoi": stoi
    }
    
    # Final report
    elapsed = time.time() - start_time
    print(f"\nFinished processing in {elapsed:.1f} seconds.")
    print(f"Total files processed: {len(all_files)}")
    print(f"Average speed: {len(all_files) / elapsed:.1f} files/sec")
    
    return tokenizer


def save_tokenizer(tokenizer: Dict, output_path: str) -> None:
    """
    Save the tokenizer to a JSON file.
    
    Args:
        tokenizer: Tokenizer dictionary
        output_path: Path to save the tokenizer JSON
    """
    # Convert int keys to strings for JSON serialization
    tokenizer_json = {
        "vocab_size": tokenizer["vocab_size"],
        "itos": {str(k): v for k, v in tokenizer["itos"].items()},
        "stoi": tokenizer["stoi"]
    }
    
    with open(output_path, 'w') as f:
        json.dump(tokenizer_json, f, indent=2)
    
    print(f"Tokenizer saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build a simplified tokenizer for MIDI encodings in parallel')
    parser.add_argument('--dirs', nargs='+', required=True, help='List of directories containing encoded MIDI files')
    parser.add_argument('--output', type=str, default='midi_tokenizer.json', help='Output path for tokenizer JSON')
    parser.add_argument('--no-recursive', action='store_true', help='Disable recursive directory scanning')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=500, 
                       help='Number of files to process in each batch (default: 100)')
    
    args = parser.parse_args()
    
    # Build tokenizer in parallel
    tokenizer = build_tokenizer_parallel(
        args.dirs, 
        recursive=not args.no_recursive,
        workers=args.workers,
        batch_size=args.batch_size
    )
    
    # Print stats
    print(f"Vocabulary size: {tokenizer['vocab_size']}")
    
    # Count different token types
    n_tokens = sum(1 for token in tokenizer["stoi"].keys() if token.startswith('n'))
    v_tokens = sum(1 for token in tokenizer["stoi"].keys() if token.startswith('v'))
    d_tokens = sum(1 for token in tokenizer["stoi"].keys() if token.startswith('d'))
    other_tokens = tokenizer["vocab_size"] - n_tokens - v_tokens - d_tokens
    
    print(f"Token distribution:")
    print(f"  n tokens: {n_tokens}")
    print(f"  v tokens: {v_tokens}")
    print(f"  d tokens: {d_tokens}")
    print(f"  other tokens: {other_tokens}")
    
    # Find highest d value
    d_values = [int(token[1:]) for token in tokenizer["stoi"].keys() if token.startswith('d')]
    if d_values:
        max_d = max(d_values)
        print(f"Highest d value: d{max_d}")
    
    # Save tokenizer
    save_tokenizer(tokenizer, args.output)


if __name__ == "__main__":
    main()