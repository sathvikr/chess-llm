#!/usr/bin/env python3
"""
Training script for chess searchless model.
Adapted from the Google DeepMind searchless chess repository.
"""

import os
import sys
from pathlib import Path
import argparse

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils

from chess_train_config import get_chess_config, get_chess_eval_config


def main():
    parser = argparse.ArgumentParser(description='Train chess searchless chess model')
    parser.add_argument('--data-path', default='chess_data', help='Path to prepared chess data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--num-return-buckets', type=int, default=128, help='Number of return buckets')
    parser.add_argument('--model-size', choices=['small', 'medium', 'large'], default='small', 
                       help='Model size configuration')
    parser.add_argument('--checkpoint-dir', default='chess_checkpoints', help='Directory for checkpoints')
    
    args = parser.parse_args()
    
    # Set up data path
    original_cwd = os.getcwd()
    os.chdir(args.data_path)
    
    try:
        # Model size configurations
        model_configs = {
            'small': {
                'num_heads': 4,
                'num_layers': 4,
                'embedding_dim': 64,
            },
            'medium': {
                'num_heads': 8,
                'num_layers': 8,
                'embedding_dim': 128,
            },
            'large': {
                'num_heads': 12,
                'num_layers': 12,
                'embedding_dim': 256,
            }
        }
        
        model_config = model_configs[args.model_size]
        
        # Get training configuration
        train_config = get_chess_config(
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            num_return_buckets=args.num_return_buckets,
        )
        
        # Configure transformer for state-value prediction
        predictor_config = transformer.TransformerConfig(
            vocab_size=len(tokenizer._CHARACTERS),  # Vocabulary size for FEN tokenization
            output_size=args.num_return_buckets,    # Predicting return buckets
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,  # FEN + return bucket
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            embedding_dim=model_config['embedding_dim'],
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=True,  # Causal mask for autoregressive prediction
        )
        
        print("Training Configuration:")
        print(f"  Data path: {args.data_path}")
        print(f"  Model size: {args.model_size}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Training steps: {args.num_steps}")
        print(f"  Model parameters:")
        print(f"    - Heads: {model_config['num_heads']}")
        print(f"    - Layers: {model_config['num_layers']}")
        print(f"    - Embedding dim: {model_config['embedding_dim']}")
        print(f"    - Vocab size: {len(tokenizer._CHARACTERS)}")
        print(f"    - Output size: {args.num_return_buckets}")
        print(f"    - Max sequence length: {tokenizer.SEQUENCE_LENGTH + 1}")
        print()
        
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Train the model
        print("Starting training...")
        params = training.train(
            train_config=train_config,
            predictor_config=predictor_config,
            build_data_loader=data_loader.build_data_loader,
        )
        
        print("Training completed!")
        
        # Optional: Run evaluation if test data exists
        test_split_path = Path(args.data_path) / "test" / "state_value_data.bag"
        if test_split_path.exists():
            print("Running evaluation on test set...")
            eval_config = get_chess_eval_config(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_return_buckets=args.num_return_buckets,
            )
            
            predictor = transformer.build_transformer_predictor(predictor_config)
            evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
            eval_results = evaluator.step(params=params, step=train_config.num_steps)
            print(f"Evaluation results: {eval_results}")
        else:
            print("No test data found, skipping evaluation.")
        
        print(f"Model training complete! Checkpoints saved to: {args.checkpoint_dir}")
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()