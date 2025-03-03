import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import argparse

@dataclass
class AugmentationStrategy:
    """Defines how to augment an encoding."""
    name: str
    pitch_shift: Optional[int] = None
    velocity_shift: Optional[int] = None
    
    def get_shifts(self) -> Tuple[int, int]:
        """Returns the pitch and velocity shifts for this strategy."""
        pitch_shift = self.pitch_shift if self.pitch_shift is not None else 0
        velocity_shift = self.velocity_shift if self.velocity_shift is not None else 0
        return pitch_shift, velocity_shift

class MIDIAugmentor:
    def __init__(self, strategies: List[AugmentationStrategy]):
        self.strategies = strategies
        self.PIANO_RANGE = (21, 108)  # Standard piano MIDI range (88 keys: A0 to C8)
        self.VELOCITY_RANGE = (0, 127)  # MIDI velocity range
        
    def _is_valid_shift(self, token: str, shift: int, shift_type: str) -> bool:
        """Check if applying the shift would result in a valid MIDI value."""
        if shift_type == 'pitch' and token.startswith('n'):
            try:
                value = int(token[1:])
                return self.PIANO_RANGE[0] <= value + shift <= self.PIANO_RANGE[1]
            except ValueError:
                return False
        elif shift_type == 'velocity' and token.startswith('v'):
            try:
                value = int(token[1:])
                return self.VELOCITY_RANGE[0] <= value + shift <= self.VELOCITY_RANGE[1]
            except ValueError:
                return False
        return True

    def _augment_token(self, token: str, pitch_shift: int, velocity_shift: int) -> str:
        """Augment a single token based on its type."""
        if token.startswith('n') and pitch_shift != 0:
            try:
                pitch = int(token[1:])
                new_pitch = max(self.PIANO_RANGE[0], min(self.PIANO_RANGE[1], pitch + pitch_shift))
                return f'n{new_pitch}'
            except ValueError:
                return token
        elif token.startswith('v') and velocity_shift != 0:
            try:
                velocity = int(token[1:])
                new_velocity = max(self.VELOCITY_RANGE[0], min(self.VELOCITY_RANGE[1], velocity + velocity_shift))
                return f'v{new_velocity}'
            except ValueError:
                return token
        return token

    def _is_valid_encoding(self, encoding: str, pitch_shift: int, velocity_shift: int) -> bool:
        """Verify that all pitch and velocity shifts stay within valid MIDI ranges."""
        tokens = encoding.split()
        for token in tokens:
            if token.startswith('n'):
                if not self._is_valid_shift(token, pitch_shift, 'pitch'):
                    return False
            elif token.startswith('v'):
                if not self._is_valid_shift(token, velocity_shift, 'velocity'):
                    return False
        return True

    def augment_encoding(self, encoding: str) -> List[Tuple[str, str]]:
        """
        Augments the encoding using each strategy.
        Returns list of (augmented_encoding, strategy_name) tuples.
        """
        variants = []
        for strategy in self.strategies:
            pitch_shift, velocity_shift = strategy.get_shifts()
            
            if not self._is_valid_encoding(encoding, pitch_shift, velocity_shift):
                continue
                
            tokens = encoding.split()
            augmented_tokens = []
            
            for token in tokens:
                augmented_token = self._augment_token(token, pitch_shift, velocity_shift)
                augmented_tokens.append(augmented_token)
                
            augmented_encoding = ' '.join(augmented_tokens)
            variants.append((augmented_encoding, strategy.name))
            
        return variants

def find_encoding_files(directories):
    """Recursively find all text encoding files in multiple directories"""
    encoding_files = []
    for directory in directories:
        encoding_files.extend([str(p) for p in Path(directory).rglob('*.txt')])
    return encoding_files

def process_encoding_file(file_path: str, output_directory: str, strategies: List[AugmentationStrategy]) -> str:
    """Process a single encoding file"""
    try:
        # Create the augmentor
        augmentor = MIDIAugmentor(strategies)
        
        # Get the filename and create output path
        file_name = Path(file_path).stem
        relative_path = Path(file_path).relative_to(Path(file_path).parent.parent)
        output_subdir = Path(output_directory) / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Check if any augmented versions already exist
        if any((output_subdir / f"{file_name}_{strategy.name}.txt").exists() 
               for strategy in strategies):
            return 'skipped'
        
        # Read the encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            encoding = f.read().strip()
        
        # Generate augmented variants
        variants = augmentor.augment_encoding(encoding)
        
        # Save each variant
        for variant, strategy_name in variants:
            output_path = output_subdir / f"{file_name}_{strategy_name}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(variant)
        
        return 'success'
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 'error'

def main():
    # Argument parsing for input and output directories and number of processes
    parser = argparse.ArgumentParser(description="MIDI Augmentation Script")
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Input directories to search for encoding files")
    parser.add_argument("--output_dir", required=True, help="Output directory for augmented encoding files")
    parser.add_argument("--num_processes", type=int, default=(os.cpu_count() - 1 or 1), help="Number of processes to use")
    args = parser.parse_args()
    
    # Define your strategies
    strategies = [
        AugmentationStrategy(name="pitch_up_1", pitch_shift=1),
        AugmentationStrategy(name="pitch_down_1", pitch_shift=-1),
        AugmentationStrategy(name="pitch_up_2", pitch_shift=2),
        AugmentationStrategy(name="pitch_down_2", pitch_shift=-2),
        AugmentationStrategy(name="pitch_up_3", pitch_shift=3),
        AugmentationStrategy(name="pitch_down_3", pitch_shift=-3),
        AugmentationStrategy(name="velocity_up_5", velocity_shift=5),
        AugmentationStrategy(name="velocity_down_5", velocity_shift=-5),
        AugmentationStrategy(name="velocity_up_10", velocity_shift=10),
        AugmentationStrategy(name="velocity_down_10", velocity_shift=-10),
        AugmentationStrategy(name="pitch_up_1_vel_up_5", pitch_shift=1, velocity_shift=5),
        AugmentationStrategy(name="pitch_down_1_vel_down_5", pitch_shift=-1, velocity_shift=-5),
        AugmentationStrategy(name="pitch_up_2_vel_up_10", pitch_shift=2, velocity_shift=10),
        AugmentationStrategy(name="pitch_down_2_vel_down_10", pitch_shift=-2, velocity_shift=-10),
        AugmentationStrategy(name="pitch_up_3_vel_up_15", pitch_shift=3, velocity_shift=15),
        AugmentationStrategy(name="pitch_down_3_vel_down_15", pitch_shift=-3, velocity_shift=-15),
        AugmentationStrategy(name="pitch_up_4_vel_up_20", pitch_shift=4, velocity_shift=20),
        AugmentationStrategy(name="pitch_down_4_vel_down_20", pitch_shift=-4, velocity_shift=-20),
    ]
    
    # Get input and output directories from arguments
    input_directories = args.input_dirs
    output_directory = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Find all encoding files
    print("Searching for encoding files...")
    encoding_files = find_encoding_files(input_directories)
    total_files = len(encoding_files)
    print(f"Found {total_files} encoding files to process")
    
    # Process files in parallel
    process_func = partial(process_encoding_file, 
                           output_directory=output_directory,
                           strategies=strategies)
    
    num_processes = args.num_processes
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Use tqdm for progress tracking
        for result in tqdm(executor.map(process_func, encoding_files), 
                         total=total_files, 
                         desc="Processing encoding files"):
            results.append(result)
    
    # Calculate statistics
    processed = results.count('success')
    skipped = results.count('skipped')
    errors = results.count('error')
    
    # Calculate timing
    total_time = time.time() - start_time
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total encoding files found: {total_files}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total processing time: {total_time:.2f} seconds")
    if processed > 0:
        print(f"Average time per file: {(total_time/processed):.2f} seconds")

if __name__ == "__main__":
    main()
