#!/usr/bin/env python3
"""
Run the exact experimental setups from the paper:
"Amortized Planning with Large-Scale Transformers: A Case Study on Chess"
arxiv:2402.04494v2

Uses the existing searchless_chess repository with paper's exact configurations.
"""

import os
import sys
from pathlib import Path
import argparse
from collections.abc import Sequence

# Add the searchless_chess src to path
sys.path.append(str(Path(__file__).parent / "searchless_chess" / "src"))

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils


# Paper's exact model configurations
PAPER_MODEL_CONFIGS = {
    "small_9M": {
        "num_heads": 8,
        "num_layers": 8,
        "embedding_dim": 256,
        "params": "~9M",
        "description": "8h8l256d"
    },
    "medium_136M": {
        "num_heads": 8, 
        "num_layers": 8,
        "embedding_dim": 1024,
        "params": "~136M",
        "description": "8h8l1024d"
    },
    "large_270M": {
        "num_heads": 8,
        "num_layers": 16,
        "embedding_dim": 1024,
        "params": "~270M", 
        "description": "8h16l1024d"
    }
}


def get_paper_transformer_config(model_size: str, policy: str, num_return_buckets: int = 128):
    """Get transformer config matching paper's exact architectures."""
    
    if model_size not in PAPER_MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(PAPER_MODEL_CONFIGS.keys())}")
    
    config = PAPER_MODEL_CONFIGS[model_size]
    
    # Determine output size based on policy
    match policy:
        case 'action_value':
            output_size = num_return_buckets
        case 'behavioral_cloning':
            output_size = utils.NUM_ACTIONS
        case 'state_value':
            output_size = num_return_buckets
    
    return transformer.TransformerConfig(
        vocab_size=len(tokenizer._CHARACTERS),  # Paper uses 32 characters for FEN tokenization
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,  # 77 + 2 = 79
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        embedding_dim=config["embedding_dim"],
        apply_post_ln=True,  # Paper uses post-normalization
        apply_qk_layernorm=False,
        use_causal_mask=True,  # Causal mask for autoregressive prediction
        widening_factor=4,  # Standard SwiGLU widening factor
    )


def get_main_experiment_config(policy: str, data_path: str = "../data"):
    """
    Main experiment configuration from paper:
    - 10M training steps
    - Batch size 4096
    - Learning rate 1e-4
    - 2.67 training epochs
    """
    num_return_buckets = 128
    
    data_config = config_lib.DataConfig(
        batch_size=4096,  # Paper's main experiment batch size
        shuffle=True,
        worker_count=8,  # Increased for large batch processing
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='train',
    )
    
    train_config = config_lib.TrainConfig(
        learning_rate=1e-4,  # Paper's main experiment learning rate
        data=data_config,
        log_frequency=1000,  # Log every 1000 steps
        num_steps=10_000_000,  # 10M steps from paper
        ckpt_frequency=100_000,  # Save checkpoint every 100k steps
        save_frequency=500_000,  # Permanent save every 500k steps
    )
    
    return train_config, num_return_buckets


def get_ablation_experiment_config(policy: str, data_path: str = "../data"):
    """
    Ablation experiment configuration from paper:
    - 5M training steps
    - Batch size 1024
    - Learning rate 4e-4
    - 3.19 training epochs
    """
    num_return_buckets = 128
    
    data_config = config_lib.DataConfig(
        batch_size=1024,  # Paper's ablation experiment batch size
        shuffle=True,
        worker_count=4,
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='train',
    )
    
    train_config = config_lib.TrainConfig(
        learning_rate=4e-4,  # Paper's ablation experiment learning rate
        data=data_config,
        log_frequency=500,  # More frequent logging for shorter runs
        num_steps=5_000_000,  # 5M steps from paper
        ckpt_frequency=50_000,  # More frequent checkpoints
        save_frequency=250_000,
    )
    
    return train_config, num_return_buckets


def get_eval_config(policy: str, num_return_buckets: int, data_path: str = "../data"):
    """Get evaluation configuration matching paper's setup."""
    
    data_config = config_lib.DataConfig(
        batch_size=256,
        shuffle=False,
        worker_count=4,
        num_return_buckets=num_return_buckets,
        policy=policy,
        split='test',
    )
    
    eval_config = config_lib.EvalConfig(
        data=data_config,
        use_ema_params=False,
        policy=policy,
        batch_size=256,
        num_return_buckets=num_return_buckets,
        num_eval_data=None,  # Evaluate on full test set
    )
    
    return eval_config


def run_experiment(model_size: str, policy: str, experiment_type: str = "main", data_path: str = "../data"):
    """Run a single experiment with paper's configuration."""
    
    print("=" * 80)
    print(f"Running {experiment_type} experiment:")
    print(f"  Model: {PAPER_MODEL_CONFIGS[model_size]['description']} ({PAPER_MODEL_CONFIGS[model_size]['params']})")
    print(f"  Policy: {policy}")
    print("=" * 80)
    
    # Get configurations
    if experiment_type == "main":
        train_config, num_return_buckets = get_main_experiment_config(policy, data_path)
    else:
        train_config, num_return_buckets = get_ablation_experiment_config(policy, data_path)
    
    predictor_config = get_paper_transformer_config(model_size, policy, num_return_buckets)
    eval_config = get_eval_config(policy, num_return_buckets, data_path)
    
    # Print configuration summary
    print(f"Training Configuration:")
    print(f"  Steps: {train_config.num_steps:,}")
    print(f"  Batch size: {train_config.data.batch_size}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Return buckets: {num_return_buckets}")
    print()
    
    print(f"Model Configuration:")
    print(f"  Heads: {predictor_config.num_heads}")
    print(f"  Layers: {predictor_config.num_layers}")
    print(f"  Embedding dim: {predictor_config.embedding_dim}")
    print(f"  Vocab size: {predictor_config.vocab_size}")
    print(f"  Output size: {predictor_config.output_size}")
    print(f"  Max sequence length: {predictor_config.max_sequence_length}")
    print()
    
    # Change to searchless_chess/src directory for training (where data loader expects to find data)
    original_cwd = os.getcwd()
    training_dir = Path(__file__).parent / "searchless_chess" / "src"
    os.chdir(training_dir)
    
    try:
        # Train the model
        print("Starting training...")
        params = training.train(
            train_config=train_config,
            predictor_config=predictor_config,
            build_data_loader=data_loader.build_data_loader,
        )
        
        print("Training completed!")
        
        # Evaluate the model
        print("Running evaluation...")
        predictor = transformer.build_transformer_predictor(predictor_config)
        evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
        eval_results = evaluator.step(params=params, step=train_config.num_steps)
        
        print("Evaluation results:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value}")
            
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    print(f"Experiment completed: {model_size} {policy} {experiment_type}")
    print("=" * 80)
    return params, eval_results


def main():
    parser = argparse.ArgumentParser(description='Run paper experiments with exact configurations')
    parser.add_argument('--model-size', choices=list(PAPER_MODEL_CONFIGS.keys()), 
                       default='small_9M', help='Model size from paper')
    parser.add_argument('--policy', choices=['action_value', 'state_value', 'behavioral_cloning'],
                       default='action_value', help='Policy type')
    parser.add_argument('--experiment-type', choices=['main', 'ablation'], default='ablation',
                       help='Experiment type (main=10M steps, ablation=5M steps)')
    parser.add_argument('--data-path', default='../data', help='Path to data directory')
    parser.add_argument('--run-all', action='store_true', help='Run all paper configurations')
    
    args = parser.parse_args()
    
    if args.run_all:
        # Run all the main configurations from the paper
        configs_to_run = [
            ("large_270M", "action_value", "main"),
            ("medium_136M", "action_value", "ablation"), 
            ("small_9M", "action_value", "ablation"),
            ("large_270M", "state_value", "ablation"),
            ("large_270M", "behavioral_cloning", "ablation"),
        ]
        
        results = {}
        for model_size, policy, exp_type in configs_to_run:
            print(f"Running configuration: {model_size} {policy} {exp_type}")
            try:
                params, eval_results = run_experiment(model_size, policy, exp_type, args.data_path)
                results[f"{model_size}_{policy}_{exp_type}"] = eval_results
            except Exception as e:
                print(f"Error in experiment {model_size} {policy} {exp_type}: {e}")
                continue
        
        # Print summary of all results
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        for config_name, result in results.items():
            print(f"{config_name}:")
            for metric, value in result.items():
                print(f"  {metric}: {value}")
            print()
    
    else:
        # Run single experiment
        run_experiment(args.model_size, args.policy, args.experiment_type, args.data_path)


if __name__ == "__main__":
    main()