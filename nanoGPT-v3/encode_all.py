import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from encoding_decoding import save_text_representation
from functools import partial
import warnings
import music21
import argparse

# Suppress music21 warnings
warnings.filterwarnings('ignore', module='music21')
# Additional specific warning suppression if needed
music21.environment.Environment()['warnings'] = 0 

def find_midi_files(directories):
    """Recursively find all MIDI files in multiple directories using Path"""
    midi_files = []
    for directory in directories:
        # Use Path().rglob() which is faster than os.walk()
        midi_files.extend([str(p) for p in Path(directory).rglob('*.mid')])
    return midi_files

def process_midi_file(midi_path, text_directory):
    """Process a single MIDI file"""
    try:
        filename = os.path.basename(midi_path)
        name = filename.split('.')[0]
        output_file = os.path.join(text_directory, f"{name}.txt")
        
        if os.path.isfile(output_file):
            return 'skipped'
            
        save_text_representation(midi_path, text_directory)
        return 'success'
        
    except Exception as e:
        return 'error'

def main():
    # Argument parsing for MIDI directories, text directory and number of processes
    parser = argparse.ArgumentParser(description="MIDI to Text Conversion Script")
    parser.add_argument("--midi_dirs", nargs="+", required=True, help="List of directories to search for MIDI files")
    parser.add_argument("--text_dir", required=True, help="Output directory to save text representations")
    parser.add_argument("--num_processes", type=int, default=(os.cpu_count() - 1 or 1), help="Number of processes to use")
    args = parser.parse_args()
    
    # Get input and output directories from arguments
    midi_directories = args.midi_dirs
    text_directory = args.text_dir
    num_processes = args.num_processes

    # Create output directory if it doesn't exist
    os.makedirs(text_directory, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Find all MIDI files
    print("Searching for MIDI files...")
    midi_files = find_midi_files(midi_directories)
    total_files = len(midi_files)
    print(f"Found {total_files} MIDI files to process")
    
    # Process files in parallel
    process_func = partial(process_midi_file, text_directory=text_directory)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Use tqdm for better progress tracking
        for result in tqdm(executor.map(process_func, midi_files), 
                         total=total_files, 
                         desc="Processing MIDI files"):
            results.append(result)
    
    # Calculate statistics
    processed = results.count('success')
    skipped = results.count('skipped')
    errors = results.count('error')
    
    # Calculate timing
    total_time = time.time() - start_time
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total MIDI files found: {total_files}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total processing time: {total_time:.2f} seconds")
    if processed > 0:
        print(f"Average time per file: {(total_time/processed):.2f} seconds")

if __name__ == "__main__":
    main()
