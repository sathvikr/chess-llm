#!/usr/bin/env python3
"""
Prepare chess evaluation data for the paper's exact experimental setup.
Converts data to the format expected by their existing codebase.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple
import argparse
import numpy as np

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import bagz
from searchless_chess.src import constants


def load_chess_evals(jsonl_path: str) -> List[Tuple[str, float]]:
    """Load chess evaluations from JSONL file and return (fen, win_prob) tuples."""
    evaluations = []
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Handle format: ["Chess: <fen>", <value>]
                    if isinstance(data, list) and len(data) == 2:
                        fen_with_prefix = data[0]
                        value = data[1]
                        # Extract FEN from "Chess: <fen>" format
                        if fen_with_prefix.startswith("Chess: "):
                            fen = fen_with_prefix[7:]  # Remove "Chess: " prefix
                            evaluations.append((fen, float(value)))
                    # Handle standard JSON object format
                    elif isinstance(data, dict) and 'fen' in data and 'value' in data:
                        evaluations.append((data['fen'], float(data['value'])))
                    else:
                        print(f"Warning: Skipping unrecognized data format at line {line_num}")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue
                except (KeyError, TypeError, ValueError) as e:
                    print(f"Warning: Data format error at line {line_num}: {e}")
                    continue
    
    return evaluations


def create_state_value_data(evaluations: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Create state-value data: (fen, win_prob)"""
    return evaluations  # Already in the right format


def create_action_value_data(evaluations: List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    """
    Create action-value data: (fen, move, win_prob)
    For this demo, we use placeholder moves since we don't have the actual game moves.
    In the real paper data, they had actual moves from game sequences.
    """
    action_value_data = []
    placeholder_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"]  # Common opening moves
    
    for i, (fen, win_prob) in enumerate(evaluations):
        # Use different placeholder moves to add some variety
        move = placeholder_moves[i % len(placeholder_moves)]
        action_value_data.append((fen, move, win_prob))
    
    return action_value_data


def create_behavioral_cloning_data(evaluations: List[Tuple[str, float]]) -> List[Tuple[str, str]]:
    """
    Create behavioral cloning data: (fen, move)
    Using placeholder moves since we don't have actual game moves.
    """
    bc_data = []
    placeholder_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"]
    
    for i, (fen, _) in enumerate(evaluations):
        move = placeholder_moves[i % len(placeholder_moves)]
        bc_data.append((fen, move))
    
    return bc_data


def split_data(data: List, train_ratio: float = 0.9, seed: int = 42) -> Tuple[List, List]:
    """Split data into train/test sets."""
    np.random.seed(seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data)
    
    # Split
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data


def write_bag_file(data: List, output_path: str, data_type: str):
    """Write data to bag file format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    coder = constants.CODERS[data_type]
    
    with bagz.BagWriter(output_path) as writer:
        for item in data:
            try:
                encoded = coder.encode(item)
                writer.write(encoded)
            except Exception as e:
                print(f"Warning: Failed to encode item {item}: {e}")
                continue
    
    print(f"Written {len(data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for paper experiments')
    parser.add_argument('--input', default='chess_evals.jsonl', help='Input JSONL file')
    parser.add_argument('--output-dir', default='searchless_chess/data', 
                       help='Output directory (should be searchless_chess/data for their scripts)')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Training data ratio')
    parser.add_argument('--create-all', action='store_true', 
                       help='Create all data formats (action_value, state_value, behavioral_cloning)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preparing Data for Paper Experiments")
    print("Using existing searchless_chess repository format")
    print("=" * 60)
    
    # Load evaluations
    print(f"Loading evaluations from {args.input}...")
    evaluations = load_chess_evals(args.input)
    
    if len(evaluations) == 0:
        print("ERROR: No valid evaluations found!")
        return
    
    print(f"Loaded {len(evaluations):,} evaluations")
    
    # Determine which formats to create
    if args.create_all:
        formats = ['state_value', 'action_value', 'behavioral_cloning']
    else:
        formats = ['state_value']  # Default to state_value which works best with our data
    
    for data_format in formats:
        print(f"\nProcessing {data_format} format...")
        
        # Create formatted data
        if data_format == 'state_value':
            formatted_data = create_state_value_data(evaluations)
        elif data_format == 'action_value':
            formatted_data = create_action_value_data(evaluations)
        elif data_format == 'behavioral_cloning':
            formatted_data = create_behavioral_cloning_data(evaluations)
        
        # Split data
        train_data, test_data = split_data(formatted_data, args.train_ratio)
        
        # Create output directories
        train_dir = os.path.join(args.output_dir, 'train')
        test_dir = os.path.join(args.output_dir, 'test')
        
        # Write train data
        train_path = os.path.join(train_dir, f'{data_format}_data.bag')
        write_bag_file(train_data, train_path, data_format)
        
        # Write test data
        test_path = os.path.join(test_dir, f'{data_format}_data.bag')
        write_bag_file(test_data, test_path, data_format)
        
        # Print statistics
        print(f"  Training set: {len(train_data):,} samples")
        print(f"  Test set: {len(test_data):,} samples")
        print(f"  Total: {len(formatted_data):,} samples")
    
    print(f"\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Data written to: {args.output_dir}")
    print("Ready to run paper experiments with:")
    print("  python run_paper_experiments.py")
    print("=" * 60)


if __name__ == "__main__":
    main()