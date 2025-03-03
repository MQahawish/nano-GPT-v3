import numpy as np
import os
import pickle
from typing import List, Dict, Any, Union, Optional, Iterator, Tuple
import time
from datetime import datetime
import json
import psutil
from pathlib import Path
import sys
import traceback
import multiprocessing
from multiprocessing import Pool
from functools import partial
import tqdm
import random
import argparse

# TokenizerWrapper class remains unchanged
class TokenizerWrapper:
    """Wrapper class to provide a unified interface for different tokenizer types"""
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = self._detect_tokenizer_type(tokenizer_path)
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.pad_token_id = self._get_pad_token_id()
        self.vocab_size = self._get_vocab_size()
        
    def _detect_tokenizer_type(self, tokenizer_path: str) -> str:
        """Detect tokenizer type based on file content."""
        ext = Path(tokenizer_path).suffix.lower()
        
        if ext == '.json':
            # Read the JSON file
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format in {tokenizer_path}")
                
                # Check for our simple tokenizer format
                if "itos" in content and "stoi" in content and "vocab_size" in content:
                    return "simple"
                
                # Check for BPE tokenizer format (HuggingFace tokenizers)
                if "model" in content and ("vocab" in content or "merges" in content):
                    return "bpe"
                
                # More detailed check for simple tokenizer
                if isinstance(content, dict):
                    # Check if content has stoi or itos keys at any level
                    if any(key in ["stoi", "itos"] for key in content.keys()):
                        return "simple"
                    
                    # Check if it has properties that look like a vocabulary mapping
                    if any(isinstance(v, dict) and len(v) > 10 for v in content.values()):
                        return "simple"
                
                raise ValueError(f"Unknown JSON tokenizer format in {tokenizer_path}")
        else:
            raise ValueError(f"Unsupported tokenizer file extension: {ext}")
    
    def _load_tokenizer(self, tokenizer_path: str) -> Any:
        """Load the appropriate tokenizer based on detected type."""
        if self.tokenizer_type == "simple":
            # Load simple tokenizer
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_dict = json.load(f)
            return tokenizer_dict
        elif self.tokenizer_type == "bpe":
            # Load BPE tokenizer using Hugging Face tokenizers
            try:
                from tokenizers import Tokenizer
                return Tokenizer.from_file(tokenizer_path)
            except ImportError:
                raise ImportError("Please install the 'tokenizers' package: pip install tokenizers")
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def _get_pad_token_id(self) -> int:
        """Get pad token ID for the loaded tokenizer."""
        if self.tokenizer_type == "simple":
            # For simple tokenizer, return 0 or check if there's a [PAD] token
            pad_token = None
            
            # Try different ways to find the pad token
            if "stoi" in self.tokenizer:
                pad_token = self.tokenizer["stoi"].get("[PAD]")
            
            if pad_token is None:
                # Return 0 as default pad token
                return 0
            return pad_token
        elif self.tokenizer_type == "bpe":
            # For BPE tokenizer, get from vocabulary
            vocab = self.tokenizer.get_vocab()
            return vocab.get("[PAD]", 0)
        return 0
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size from the tokenizer."""
        if self.tokenizer_type == "simple":
            # Try to get vocab_size from tokenizer dict
            if "vocab_size" in self.tokenizer:
                return self.tokenizer["vocab_size"]
            # Otherwise, compute from stoi 
            elif "stoi" in self.tokenizer:
                return len(self.tokenizer["stoi"])
            # Try to compute from itos
            elif "itos" in self.tokenizer:
                return len(self.tokenizer["itos"])
            return 0
        elif self.tokenizer_type == "bpe":
            return self.tokenizer.get_vocab_size()
        return 0
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with unified interface."""
        if self.tokenizer_type == "simple":
            # Simple tokenizer encoding
            text = text.replace('sepxx', ' sepxx ')  # Tokenization preprocessing
            tokens = text.strip().split()
            # Convert tokens to IDs using stoi mapping
            stoi = self.tokenizer.get("stoi", {})
            return [stoi.get(token, 0) for token in tokens]
        elif self.tokenizer_type == "bpe":
            # BPE tokenizer encoding
            encoding = self.tokenizer.encode(text)
            return encoding.ids
        return []

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text with unified interface."""
        if self.tokenizer_type == "simple":
            # Simple tokenizer decoding
            itos = {}
            
            # Handle different formats of itos
            raw_itos = self.tokenizer.get("itos", {})
            for k, v in raw_itos.items():
                # Convert keys to int if they're stored as strings
                try:
                    key = int(k)
                except (ValueError, TypeError):
                    key = k
                itos[key] = v
                
            tokens = [itos.get(id, "[UNK]") for id in ids]
            text = " ".join(tokens)
            text = text.replace(" sepxx ", " sepxx")
            return text
        elif self.tokenizer_type == "bpe":
            # BPE tokenizer decoding
            return self.tokenizer.decode(ids)
        return ""


# Function to process a single file - made standalone for multiprocessing
def process_single_file(file_path: str, tokenizer: TokenizerWrapper) -> Tuple[str, List[int]]:
    """Process a single file and return its path and tokens."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tokens = tokenizer.encode(content)
        return file_path, tokens
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, []


class SequencePreparation:
    def __init__(self, main_data_path: str, augmented_data_path: str, 
                 output_base_dir: str, tokenizer_path: str,
                 augmented_percentage: float = 100.0,
                 fixed_stride: int = 256, batch_size: int = 100000, 
                 num_workers: int = None, seed: int = 42):
        self.main_data_path = main_data_path
        self.augmented_data_path = augmented_data_path
        self.augmented_percentage = augmented_percentage
        self.output_base_dir = output_base_dir
        self.tokenizer_path = tokenizer_path
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer_path)
        self.tokenizer_type = self.tokenizer_wrapper.tokenizer_type
        self.pad_token_id = self.tokenizer_wrapper.pad_token_id
        self.fixed_stride = fixed_stride
        self.batch_size = batch_size  # Number of sequences to process at once
        self.seed = seed  # Random seed for reproducibility
        
        # Set number of workers based on CPU cores
        if num_workers is None:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = num_workers
            
        print(f"Using tokenizer type: {self.tokenizer_type}")
        print(f"Using special tokens - PAD: {self.pad_token_id}")
        print(f"Vocabulary size: {self.tokenizer_wrapper.vocab_size}")
        print(f"Batch size: {self.batch_size} sequences")
        print(f"Number of parallel workers: {self.num_workers}")
        print(f"Augmented dataset percentage: {self.augmented_percentage}%")
        
        # Extract tokenizer name from path and create specific output directory
        tokenizer_name = Path(tokenizer_path).stem
        # Include percentage in the output directory name if sampling
        if augmented_percentage < 100:
            self.output_dir = os.path.join(output_base_dir, f"{tokenizer_name}_aug{int(augmented_percentage)}")
        else:
            self.output_dir = os.path.join(output_base_dir, tokenizer_name)
        
        # Keep your original sequence lengths
        self.sequence_configs = {
            "tiny": 512,  
            "small": 1024,  
            # "medium": 2048,    
            # "big": 4096,
        }
        
        self.stats = {}
        self.sequence_analysis = {}
        
    def count_eligible_songs(self, file_tokens_map: Dict[str, List[int]], max_length: int, stride: int = None) -> int:
        """Count total number of sequences that will be generated with strided windows."""
        if stride is None:
            stride = self.fixed_stride
            
        total_sequences = 0
        for tokens in file_tokens_map.values():
            if len(tokens) <= max_length:
                total_sequences += 1
            else:
                # Calculate number of full windows with stride
                num_windows = (len(tokens) - max_length) // stride + 1
                total_sequences += num_windows
                # Add one more if there's a remainder that needs handling
                if (len(tokens) - max_length) % stride != 0:
                    total_sequences += 1
        
        return total_sequences

    def sequence_generator(self, file_tokens_map: Dict[str, List[int]], max_length: int) -> Iterator[np.ndarray]:
        """Generate sequences one by one using a fixed stride window approach."""
        for file_path, tokens in file_tokens_map.items():
            song_length = len(tokens)
            
            if song_length <= max_length:
                # If song is shorter than or equal to max_length, pad it
                sequence = np.array(tokens, dtype=np.uint16)
                if len(sequence) < max_length:
                    padding = np.full(max_length - len(sequence), self.pad_token_id, dtype=np.uint16)
                    sequence = np.concatenate([padding, sequence])
                yield sequence
            else:
                # Create sequences with fixed stride
                for start_idx in range(0, song_length - max_length + 1, self.fixed_stride):
                    window = tokens[start_idx:start_idx + max_length]
                    yield np.array(window, dtype=np.uint16)
                
                # Handle the last window if it doesn't align perfectly with stride
                last_start = (song_length - max_length) // self.fixed_stride * self.fixed_stride
                if last_start + max_length < song_length:
                    last_window = tokens[-max_length:]
                    yield np.array(last_window, dtype=np.uint16)

    def create_sequences_batched(self, file_tokens_map: Dict[str, List[int]], max_length: int) -> tuple:
        """Create sequences using batched processing to save memory."""
        print(f"\nCreating sequences with window length {max_length} and fixed stride {self.fixed_stride}")
        
        stats = {
            'total_songs': len(file_tokens_map),
            'songs_included': len(file_tokens_map),
            'total_sequences': 0,
            'max_song_length': 0,
            'min_song_length': float('inf'),
            'avg_song_length': 0
        }
        
        # Calculate stats first (this is memory efficient)
        total_length = 0
        for tokens in file_tokens_map.values():
            song_length = len(tokens)
            total_length += song_length
            stats['max_song_length'] = max(stats['max_song_length'], song_length)
            stats['min_song_length'] = min(stats['min_song_length'], song_length)
            
            # Count total sequences
            if song_length <= max_length:
                stats['total_sequences'] += 1
            else:
                num_windows = (song_length - max_length) // self.fixed_stride + 1
                stats['total_sequences'] += num_windows
                if (song_length - max_length) % self.fixed_stride != 0:
                    stats['total_sequences'] += 1
        
        stats['avg_song_length'] = total_length / len(file_tokens_map) if file_tokens_map else 0
        
        print(f"\nSequence Creation Stats:")
        print(f"Songs included: {stats['songs_included']}/{stats['total_songs']} (100%)")
        print(f"Total sequences to create: {stats['total_sequences']}")
        print(f"Max song length: {stats['max_song_length']}")
        print(f"Min song length: {stats['min_song_length']}")
        print(f"Average song length: {stats['avg_song_length']:.1f}")
        
        # Return the stats and a generator for the sequences
        return self.sequence_generator(file_tokens_map, max_length), stats

    def collect_files(self) -> Tuple[List[str], List[str]]:
        """
        Collect files from both main and augmented directories, 
        sampling a percentage of augmented files.
        Returns a tuple of (main_files, sampled_augmented_files)
        """
        # Collect all main files (we always use 100% of these)
        main_files = []
        print(f"\nCollecting files recursively from {self.main_data_path}")
        for root, _, files in os.walk(self.main_data_path):
            txt_files = [os.path.join(root, f) for f in files if f.endswith('.txt')]
            main_files.extend(txt_files)
            if txt_files:
                print(f"Found {len(txt_files)} files in {root}")
        
        # Collect augmented files
        augmented_files = []
        print(f"\nCollecting files recursively from {self.augmented_data_path}")
        for root, _, files in os.walk(self.augmented_data_path):
            txt_files = [os.path.join(root, f) for f in files if f.endswith('.txt')]
            augmented_files.extend(txt_files)
            if txt_files:
                print(f"Found {len(txt_files)} files in {root}")
        
        # Shuffle and sample augmented files if needed
        random.seed(self.seed)
        if self.augmented_percentage < 100:
            random.shuffle(augmented_files)
            sample_size = int(len(augmented_files) * (self.augmented_percentage / 100.0))
            augmented_files = augmented_files[:sample_size]
            print(f"\nSampling {sample_size} files ({self.augmented_percentage}%) from the augmented dataset")
        else:
            print(f"\nUsing all {len(augmented_files)} files from the augmented dataset")
        
        print(f"Total main files: {len(main_files)}")
        print(f"Total augmented files (after sampling): {len(augmented_files)}")
        
        return main_files, augmented_files

    def process_files(self) -> Dict[str, List[int]]:
        """
        Process files from both main and augmented datasets,
        using parallel processing for efficiency.
        """
        start_time = time.time()
        
        # Collect files from both datasets
        main_files, augmented_files = self.collect_files()
        
        # Combine into a single list for processing
        all_file_paths = main_files + augmented_files
        total = len(all_file_paths)
        
        print(f"\nTotal files to process: {total}")
        print(f"Processing files using {self.num_workers} workers...")
        
        # Create a partial function with the tokenizer
        process_func = partial(process_single_file, tokenizer=self.tokenizer_wrapper)
        
        # Use multiprocessing to process files in parallel
        file_tokens_map = {}
        
        # Process in chunks to avoid memory issues with very large datasets
        chunk_size = 10000  # Adjust based on your system
        
        for i in range(0, len(all_file_paths), chunk_size):
            chunk = all_file_paths[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(all_file_paths) + chunk_size - 1)//chunk_size} ({len(chunk)} files)")
            
            with Pool(processes=self.num_workers) as pool:
                # Use tqdm for progress bar
                results = list(tqdm.tqdm(
                    pool.imap(process_func, chunk),
                    total=len(chunk),
                    desc="Processing files"
                ))
            
            # Add results to the map
            for file_path, tokens in results:
                if tokens:  # Only add if tokens were successfully extracted
                    file_tokens_map[file_path] = tokens
            
            # Report progress
            print(f"Processed {min(i+chunk_size, total)}/{total} files ({min(i+chunk_size, total)/total*100:.1f}%)")
            print(f"Current memory usage: {self.get_memory_usage()}")
        
        processing_time = time.time() - start_time
        print(f"Successfully processed {len(file_tokens_map)} files out of {total}")
        print(f"File processing completed in {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        return file_tokens_map

    def prepare_all_sequences(self):
        """Prepare datasets for all sequence lengths."""
        print(f"\n{'='*80}")
        print(f"Starting sequence preparation at {datetime.now()}")
        print(f"Output directory: {self.output_dir}")
        print(f"Tokenizer type: {self.tokenizer_type}")
        print(f"Augmented dataset percentage: {self.augmented_percentage}%")
        print(f"{'='*80}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Process all files to get tokens
        start_time = time.time()
        file_tokens_map = self.process_files()
        processing_time = time.time() - start_time
        print(f"File processing completed in {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        
        total_songs = len(file_tokens_map)
        
        # Split files into train and validation (90/10 split)
        all_files = list(file_tokens_map.keys())
        random.seed(self.seed)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * 0.9)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        train_map = {f: file_tokens_map[f] for f in train_files}
        val_map = {f: file_tokens_map[f] for f in val_files}
        
        for config_name, seq_length in self.sequence_configs.items():
            print(f"\n{'='*80}")
            print(f"Configuration: {config_name} (length: {seq_length})")
            
            # Count eligible songs
            eligible_songs = self.count_eligible_songs(file_tokens_map, seq_length)
            print(f"\nThere are {eligible_songs} sequences that will be generated from {total_songs} files with length <= {seq_length} tokens")
            
            print(f"\nProceeding with {config_name} configuration preparation")
            print(f"{'='*80}")
            
            output_dir = os.path.join(self.output_dir, f"seq_{seq_length}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get sequence generators and stats
            train_seq_gen, train_stats = self.create_sequences_batched(train_map, seq_length)
            val_seq_gen, val_stats = self.create_sequences_batched(val_map, seq_length)
            
            # Save sequences in batches
            self.save_sequences_batched(train_seq_gen, val_seq_gen, output_dir, seq_length, 
                                      train_stats, val_stats)
            
            # Update statistics
            self.stats[str(seq_length)] = {
                'config_name': config_name,
                'sequence_length': seq_length,
                'train_stats': train_stats,
                'val_stats': val_stats
            }
        
        self.save_statistics()

    def save_sequences_batched(self, train_seq_gen: Iterator[np.ndarray], val_seq_gen: Iterator[np.ndarray],
                             output_dir: str, seq_length: int, train_stats: dict, val_stats: dict):
        """Save sequences in batches to avoid memory issues."""
        start_time = time.time()
        
        # Prepare output files
        train_path = os.path.join(output_dir, 'train.bin')
        val_path = os.path.join(output_dir, 'val.bin')
        
        # Process and save training sequences in batches
        print(f"\nProcessing {train_stats['total_sequences']} training sequences in batches of {self.batch_size}...")
        train_batches_processed = 0
        train_sequences_processed = 0
        
        with open(train_path, 'wb') as train_file:
            batch = []
            for sequence in train_seq_gen:
                batch.append(sequence)
                train_sequences_processed += 1
                
                if len(batch) >= self.batch_size:
                    # Process and save this batch
                    batch_array = np.stack(batch)
                    batch_array.tofile(train_file)
                    
                    train_batches_processed += 1
                    print(f"Saved training batch {train_batches_processed}, "
                          f"total sequences: {train_sequences_processed}/{train_stats['total_sequences']} "
                          f"({train_sequences_processed/train_stats['total_sequences']*100:.1f}%)")
                    
                    # Clear batch to free memory
                    batch = []
            
            # Save any remaining sequences
            if batch:
                batch_array = np.stack(batch)
                batch_array.tofile(train_file)
                train_batches_processed += 1
                print(f"Saved final training batch, "
                      f"total sequences: {train_sequences_processed}/{train_stats['total_sequences']} "
                      f"({train_sequences_processed/train_stats['total_sequences']*100:.1f}%)")
        
        # Process and save validation sequences in batches
        print(f"\nProcessing {val_stats['total_sequences']} validation sequences in batches of {self.batch_size}...")
        val_batches_processed = 0
        val_sequences_processed = 0
        
        with open(val_path, 'wb') as val_file:
            batch = []
            for sequence in val_seq_gen:
                batch.append(sequence)
                val_sequences_processed += 1
                
                if len(batch) >= self.batch_size:
                    # Process and save this batch
                    batch_array = np.stack(batch)
                    batch_array.tofile(val_file)
                    
                    val_batches_processed += 1
                    print(f"Saved validation batch {val_batches_processed}, "
                          f"total sequences: {val_sequences_processed}/{val_stats['total_sequences']} "
                          f"({val_sequences_processed/val_stats['total_sequences']*100:.1f}%)")
                    
                    # Clear batch to free memory
                    batch = []
            
            # Save any remaining sequences
            if batch:
                batch_array = np.stack(batch)
                batch_array.tofile(val_file)
                val_batches_processed += 1
                print(f"Saved final validation batch, "
                      f"total sequences: {val_sequences_processed}/{val_stats['total_sequences']} "
                      f"({val_sequences_processed/val_stats['total_sequences']*100:.1f}%)")
        
        # Save metadata
        meta = {
            'vocab_size': self.tokenizer_wrapper.vocab_size,
            'sequence_length': seq_length,
            'tokenizer_type': self.tokenizer_type,
            'augmented_percentage': self.augmented_percentage,
            'train_stats': {
                **train_stats,
                'sequences_processed': train_sequences_processed,
                'batches_processed': train_batches_processed
            },
            'val_stats': {
                **val_stats,
                'sequences_processed': val_sequences_processed,
                'batches_processed': val_batches_processed
            },
            'creation_date': datetime.now().isoformat(),
            'tokenizer_path': self.tokenizer_path,
            'special_tokens': {
                'pad': self.pad_token_id
            }
        }
        
        with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
            
        print(f"\nSaved to {output_dir}:")
        print(f"Train sequences: {train_sequences_processed}")
        print(f"Val sequences: {val_sequences_processed}")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    def save_statistics(self):
        """Save overall statistics."""
        stats_path = os.path.join(self.output_dir, 'preparation_stats.json')
        
        stats = {
            'preparation_date': datetime.now().isoformat(),
            'augmented_percentage': self.augmented_percentage,
            'configurations': self.stats,
            'tokenizer_path': self.tokenizer_path,
            'tokenizer_type': self.tokenizer_type,
            'system_info': {
                'final_memory_usage': self.get_memory_usage(),
                'python_version': sys.version
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"\nStatistics saved to {stats_path}")

    @staticmethod
    def get_memory_usage() -> str:
        """Get current memory usage."""
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024
        return f"{mem:.2f}MB"


def main():
    # Argument parsing for configuration values
    parser = argparse.ArgumentParser(description="Sequence Preparation Script")
    parser.add_argument("--main_data_path", type=str, required=True,
                        help="Path to the main dataset directory containing text files")
    parser.add_argument("--augmented_data_path", type=str, required=True,
                        help="Path to the augmented dataset directory containing text files")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base directory to save output sequences and metadata")
    parser.add_argument("--augmented_percentage", type=float, default=30.0,
                        help="Percentage of augmented files to use (e.g., 30.0 for 30%)")
    parser.add_argument("--tokenizer_paths", nargs="+", required=True,
                        help="List of tokenizer file paths to process")
    parser.add_argument("--batch_size", type=int, default=100000,
                        help="Batch size for saving sequences")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for parallel processing (default uses available cores minus one)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    try:
        for tokenizer_path in args.tokenizer_paths:
            print(f"\n{'='*80}")
            print(f"Processing with tokenizer: {tokenizer_path}")
            print(f"Using {args.augmented_percentage}% of augmented dataset")
            print(f"{'='*80}")
            prep = SequencePreparation(
                main_data_path=args.main_data_path,
                augmented_data_path=args.augmented_data_path,
                output_base_dir=args.output_base_dir,
                tokenizer_path=tokenizer_path,
                augmented_percentage=args.augmented_percentage,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed
            )
            prep.prepare_all_sequences()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
