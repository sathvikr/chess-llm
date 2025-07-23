#!/usr/bin/env python3
"""
Setup script for chess training environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return None


def main():
    print("Setting up chess training environment...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install requirements from the searchless chess repo
    requirements_path = Path(__file__).parent / "searchless_chess" / "requirements.txt"
    if requirements_path.exists():
        print(f"\nInstalling requirements from {requirements_path}...")
        cmd = f"{sys.executable} -m pip install -r {requirements_path}"
        run_command(cmd, "Installing Python dependencies")
    else:
        print("❌ Requirements file not found. Make sure searchless_chess repository is cloned.")
        sys.exit(1)
    
    # Install additional dependencies for chess
    print("\nInstalling additional dependencies...")
    additional_deps = [
        "click",  # For command line interface
        "python-chess",  # For chess board handling
    ]
    
    for dep in additional_deps:
        cmd = f"{sys.executable} -m pip install {dep}"
        run_command(cmd, f"Installing {dep}")
    
    # Check for JAX with GPU support (optional)
    print("\nChecking JAX installation...")
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        print(f"✓ JAX devices: {jax.devices()}")
        if len(jax.devices('gpu')) > 0:
            print(f"✓ GPU devices available: {len(jax.devices('gpu'))}")
        else:
            print("ℹ No GPU devices found. Training will use CPU (slower).")
            print("  To install JAX with GPU support, see: https://jax.readthedocs.io/en/latest/installation.html")
    except ImportError:
        print("❌ JAX not properly installed")
        sys.exit(1)
    
    # Verify data files
    print("\nChecking data files...")
    chess_evals = Path(__file__).parent / "chess_evals.jsonl"
    if chess_evals.exists():
        print(f"✓ Found chess evaluations: {chess_evals}")
        
        # Count lines in the file
        with open(chess_evals, 'r') as f:
            num_lines = sum(1 for _ in f)
        print(f"  - Contains {num_lines:,} evaluation records")
    else:
        print("⚠ chess_evals.jsonl not found")
        print("  Make sure to run your evaluation script first to generate training data")
    
    # Make scripts executable
    print("\nMaking scripts executable...")
    scripts = [
        "prepare_chess_data.py",
        "train_chess.py",
        "chess_train_config.py"
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            os.chmod(script_path, 0o755)
            print(f"✓ Made {script} executable")
    
    print("\n" + "=" * 50)
    print("Setup complete! 🎉")
    print("\nNext steps:")
    print("1. Prepare your training data:")
    print("   python prepare_chess_data.py --input chess_evals.jsonl")
    print()
    print("2. Start training:")
    print("   python train_chess.py --data-path chess_data")
    print()
    print("Optional parameters for training:")
    print("   --batch-size 32          # Batch size")
    print("   --learning-rate 1e-4     # Learning rate")
    print("   --num-steps 10000        # Training steps")
    print("   --model-size small       # Model size (small/medium/large)")
    print()


if __name__ == "__main__":
    main()