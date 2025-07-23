#!/usr/bin/env python3
"""
ChessBench model configurations following the exact setup from:
"Amortized Planning with Large-Scale Transformers: A Case Study on Chess"
arxiv:2402.04494v2
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import config as config_lib
from searchless_chess.src import transformer


@dataclass
class ChessBenchModelConfig:
    """Model configuration matching the paper's architectures."""
    name: str
    num_heads: int
    num_layers: int
    embedding_dim: int
    approximate_params: str  # For reporting


# Paper's exact model configurations
CHESSBENCH_MODEL_CONFIGS = {
    # Main paper configurations
    "small": ChessBenchModelConfig(
        name="8h8l256d",
        num_heads=8,
        num_layers=8, 
        embedding_dim=256,
        approximate_params="9M"
    ),
    "medium": ChessBenchModelConfig(
        name="8h8l1024d", 
        num_heads=8,
        num_layers=8,
        embedding_dim=1024,
        approximate_params="136M"
    ),
    "large": ChessBenchModelConfig(
        name="8h16l1024d",
        num_heads=8,
        num_layers=16,
        embedding_dim=1024,
        approximate_params="270M"
    )
}


def get_chessbench_transformer_config(
    model_size: str,
    policy_type: str,
    num_return_buckets: int = 128,
    vocab_size: int = 32  # Paper uses 32 for FEN tokenization
) -> transformer.TransformerConfig:
    """Get transformer config matching paper's architecture."""
    
    if model_size not in CHESSBENCH_MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(CHESSBENCH_MODEL_CONFIGS.keys())}")
    
    model_config = CHESSBENCH_MODEL_CONFIGS[model_size]
    
    # Output size depends on policy type
    if policy_type == 'action_value':
        output_size = num_return_buckets
    elif policy_type == 'state_value':
        output_size = num_return_buckets
    elif policy_type == 'behavioral_cloning':
        output_size = 4096  # Number of possible moves (paper's approximation)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    # Paper uses sequence length of 79 (77 for FEN + 2 for prediction targets)
    max_sequence_length = 79
    
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=max_sequence_length,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        embedding_dim=model_config.embedding_dim,
        apply_post_ln=True,  # Paper uses post-normalization
        apply_qk_layernorm=False,
        use_causal_mask=True,  # Causal mask for autoregressive prediction
        widening_factor=4,  # Standard transformer widening factor
    )


def get_chessbench_main_experiment_config(
    model_size: str = "large",
    policy_type: str = "action_value",
    data_path: str = "chessbench_data"
) -> config_lib.TrainConfig:
    """
    Get training configuration for main experiments from the paper.
    
    Main experiment settings:
    - 10M training steps
    - Batch size 4096
    - Learning rate 1e-4
    - Adam optimizer
    """
    
    data_config = config_lib.DataConfig(
        batch_size=4096,  # Paper's main experiment batch size
        shuffle=True,
        seed=42,
        drop_remainder=True,
        worker_count=8,  # Increased for large batch processing
        num_return_buckets=128,  # Paper's default
        split="train",
        policy=policy_type,
        num_records=None,  # Use all available data
    )
    
    train_config = config_lib.TrainConfig(
        data=data_config,
        learning_rate=1e-4,  # Paper's main experiment LR
        max_grad_norm=1.0,
        num_steps=10_000_000,  # 10M steps from paper
        ckpt_frequency=100_000,  # Save every 100k steps
        ckpt_max_to_keep=10,
        save_frequency=500_000,  # Permanent save every 500k steps
        log_frequency=1000,  # Log every 1k steps
    )
    
    return train_config


def get_chessbench_ablation_config(
    model_size: str = "small",
    policy_type: str = "action_value", 
    data_path: str = "chessbench_data"
) -> config_lib.TrainConfig:
    """
    Get training configuration for ablation experiments from the paper.
    
    Ablation experiment settings:
    - 5M training steps
    - Batch size 1024  
    - Learning rate 4e-4
    - Adam optimizer
    """
    
    data_config = config_lib.DataConfig(
        batch_size=1024,  # Paper's ablation experiment batch size
        shuffle=True,
        seed=42,
        drop_remainder=True,
        worker_count=4,
        num_return_buckets=128,
        split="train",
        policy=policy_type,
        num_records=None,
    )
    
    train_config = config_lib.TrainConfig(
        data=data_config,
        learning_rate=4e-4,  # Paper's ablation experiment LR
        max_grad_norm=1.0,
        num_steps=5_000_000,  # 5M steps from paper
        ckpt_frequency=50_000,  # More frequent checkpoints for shorter training
        ckpt_max_to_keep=20,
        save_frequency=250_000,
        log_frequency=500,
    )
    
    return train_config


def get_chessbench_eval_config(
    model_size: str = "large",
    policy_type: str = "action_value",
    data_path: str = "chessbench_data"
) -> config_lib.EvalConfig:
    """Get evaluation configuration matching paper's setup."""
    
    data_config = config_lib.DataConfig(
        batch_size=256,  # Smaller batch size for evaluation
        shuffle=False,
        seed=42,
        drop_remainder=False,
        worker_count=4,
        num_return_buckets=128,
        split="test",
        policy=policy_type,
        num_records=None,
    )
    
    eval_config = config_lib.EvalConfig(
        data=data_config,
        num_eval_data=None,  # Evaluate on full test set
        use_ema_params=False,
        policy=policy_type,
        num_return_buckets=128,
        batch_size=256,
    )
    
    return eval_config


def print_chessbench_config_summary(model_size: str, experiment_type: str = "main"):
    """Print configuration summary matching paper's reporting style."""
    model_config = CHESSBENCH_MODEL_CONFIGS[model_size]
    
    print("=" * 60)
    print("ChessBench Configuration Summary")
    print("Following arxiv:2402.04494v2")
    print("=" * 60)
    print(f"Model: {model_config.name} (~{model_config.approximate_params} parameters)")
    print(f"  Heads: {model_config.num_heads}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Embedding dim: {model_config.embedding_dim}")
    print(f"  Architecture: Vanilla decoder-only transformer")
    print(f"  Activation: SwiGLU")
    print(f"  Normalization: Post-layer normalization")
    
    if experiment_type == "main":
        print(f"\nMain Experiment Settings:")
        print(f"  Training steps: 10,000,000")
        print(f"  Batch size: 4,096")
        print(f"  Learning rate: 1e-4")
        print(f"  Training epochs: ~2.67 (estimated)")
    else:
        print(f"\nAblation Experiment Settings:")
        print(f"  Training steps: 5,000,000") 
        print(f"  Batch size: 1,024")
        print(f"  Learning rate: 4e-4")
        print(f"  Training epochs: ~3.19 (estimated)")
    
    print(f"\nData Configuration:")
    print(f"  Value buckets: 128")
    print(f"  Optimizer: Adam")
    print(f"  Sequence length: 79 tokens")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    for size in ["small", "medium", "large"]:
        print_chessbench_config_summary(size, "main")
        print()