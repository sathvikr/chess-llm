#!/usr/bin/env python3
"""
Quick test of the paper experiment setup with minimal steps.
"""

import os
import sys
from pathlib import Path

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils


def run_quick_test():
    """Run a quick test with minimal configuration."""
    
    print("=" * 60)
    print("Quick Test of Paper Experiment Setup")
    print("=" * 60)
    
    # Small test configuration
    num_return_buckets = 128
    policy = 'state_value'
    
    # Test data config
    data_config = config_lib.DataConfig(
        batch_size=32,  # Small batch for quick test
        shuffle=True,
        worker_count=0,  # No multiprocessing for test
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='train',
    )
    
    # Test training config
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,
        data=data_config,
        log_frequency=1,
        num_steps=20,  # Just 20 steps for test
        ckpt_frequency=10,
        save_frequency=20,
    )
    
    # Small model config
    predictor_config = transformer.TransformerConfig(
        vocab_size=len(tokenizer._CHARACTERS),
        output_size=num_return_buckets,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=4,  # Small model
        num_layers=2,  # Small model  
        embedding_dim=64,  # Small model
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=True,
        widening_factor=4,
    )
    
    print("Test Configuration:")
    print(f"  Steps: {train_config.num_steps}")
    print(f"  Batch size: {train_config.data.batch_size}")
    print(f"  Model: {predictor_config.num_heads}h{predictor_config.num_layers}l{predictor_config.embedding_dim}d")
    print(f"  Policy: {policy}")
    print()
    
    # Change to searchless_chess/src directory
    original_cwd = os.getcwd()
    training_dir = Path(__file__).parent / "searchless_chess" / "src"
    os.chdir(training_dir)
    
    try:
        print("Starting training...")
        params = training.train(
            train_config=train_config,
            predictor_config=predictor_config,
            build_data_loader=data_loader.build_data_loader,
        )
        
        print("✓ Training completed successfully!")
        print("Note: Skipping evaluation for quick test.")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    finally:
        os.chdir(original_cwd)
    
    print()
    print("=" * 60)
    print("✓ Quick test PASSED! Setup is working correctly.")
    print("Ready to run full paper experiments.")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()