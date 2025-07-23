#!/usr/bin/env python3
"""
Training configuration for chess searchless model.
Adapted from the Google DeepMind searchless chess repository.
"""

import sys
from pathlib import Path

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import config as config_lib


def get_chess_config(
    data_path: str = "chess_data",
    batch_size: int = 32,  # Smaller batch size for potentially smaller dataset
    learning_rate: float = 1e-4,
    num_steps: int = 10000,  # Adjust based on dataset size
    num_return_buckets: int = 128,
    split: str = "train",
) -> config_lib.TrainConfig:
    """Get training configuration for chess model."""
    
    data_config = config_lib.DataConfig(
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        drop_remainder=True,
        worker_count=4,
        num_return_buckets=num_return_buckets,
        split=split,
        policy="state_value",  # Using state-value training
        num_records=None,  # Use all available data
    )
    
    train_config = config_lib.TrainConfig(
        data=data_config,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        num_steps=num_steps,
        ckpt_frequency=1000,  # Save checkpoint every 1000 steps
        ckpt_max_to_keep=5,
        save_frequency=5000,  # Permanent save every 5000 steps
        log_frequency=100,   # Log every 100 steps
    )
    
    return train_config


def get_chess_eval_config(
    data_path: str = "chess_data",
    batch_size: int = 32,
    num_return_buckets: int = 128,
    split: str = "test",
) -> config_lib.EvalConfig:
    """Get evaluation configuration for chess model."""
    
    data_config = config_lib.DataConfig(
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        seed=42,
        drop_remainder=False,
        worker_count=4,
        num_return_buckets=num_return_buckets,
        split=split,
        policy="state_value",
        num_records=None,
    )
    
    eval_config = config_lib.EvalConfig(
        data=data_config,
        num_eval_data=None,  # Evaluate on all data
        use_ema_params=False,
        policy="state_value",
        num_return_buckets=num_return_buckets,
        batch_size=batch_size,
    )
    
    return eval_config


if __name__ == "__main__":
    # Example usage
    config = get_chess_config()
    print("Chess training configuration:")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Number of steps: {config.num_steps}")
    print(f"  Policy: {config.data.policy}")
    print(f"  Return buckets: {config.data.num_return_buckets}")