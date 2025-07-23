#!/usr/bin/env python3
"""
Prepare chess evaluation data for training with the searchless chess model.
Converts JSONL evaluation data to the format expected by the model.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import bagz
from searchless_chess.src import constants


def load_chess_evals(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load chess evaluations from JSONL file."""
    evaluations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Handle format: ["Chess: <fen>", <value>]
                if isinstance(data, list) and len(data) == 2:
                    fen_with_prefix = data[0]
                    value = data[1]
                    # Extract FEN from "Chess: <fen>" format
                    if fen_with_prefix.startswith("Chess: "):
                        fen = fen_with_prefix[7:]  # Remove "Chess: " prefix
                        evaluations.append({
                            'fen': fen,
                            'value': value,
                            'variant': 'chess'
                        })
                # Handle standard JSON object format
                elif isinstance(data, dict) and 'fen' in data and 'value' in data:
                    evaluations.append(data)
                else:
                    print(f"Warning: Skipping unrecognized data format: {data}")
    return evaluations


def convert_to_state_value_format(evaluations: List[Dict[str, Any]]) -> List[tuple]:
    """Convert evaluations to state-value format (fen, win_prob)."""
    state_value_data = []
    for eval_data in evaluations:
        fen = eval_data['fen']
        value = eval_data['value']  # Already in [0,1] range from logistic transform
        state_value_data.append((fen, value))
    return state_value_data


def write_bag_file(data: List[tuple], output_path: str, data_type: str):
    """Write data to bag file format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    coder = constants.CODERS[data_type]
    
    with bagz.BagWriter(output_path) as writer:
        for item in data:
            encoded = coder.encode(item)
            writer.write(encoded)
    
    print(f"Written {len(data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare chess data for searchless chess training')
    parser.add_argument('--input', default='chess_evals.jsonl', help='Input JSONL file with evaluations')
    parser.add_argument('--output-dir', default='chess_data', help='Output directory for bag files')
    parser.add_argument('--split', default='train', help='Data split name (train/test/val)')
    
    args = parser.parse_args()
    
    print(f"Loading evaluations from {args.input}...")
    evaluations = load_chess_evals(args.input)
    print(f"Loaded {len(evaluations)} evaluations")
    
    # Convert to state-value format
    print("Converting to state-value format...")
    state_value_data = convert_to_state_value_format(evaluations)
    
    # Create output directory structure
    split_dir = os.path.join(args.output_dir, args.split)
    
    # Write state-value data
    state_value_path = os.path.join(split_dir, 'state_value_data.bag')
    write_bag_file(state_value_data, state_value_path, 'state_value')
    
    print(f"\nData preparation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Records: {len(state_value_data)}")
    print(f"\nTo train the model, update the data path in the training config to point to: {args.output_dir}")


if __name__ == "__main__":
    main()