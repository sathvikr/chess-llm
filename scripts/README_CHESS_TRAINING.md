# Chess Training with Searchless Chess Model

This setup adapts the Google DeepMind searchless chess model for training on regular chess games. The model uses a transformer architecture to learn chess positions without explicit search.

## Overview

The training pipeline includes:
1. **Data Preparation**: Convert your chess evaluations to the model's expected format
2. **Model Architecture**: Transformer-based predictor for chess
3. **Training**: State-value prediction using your evaluation data
4. **Evaluation**: Test the trained model's performance

## Quick Start

### 1. Setup Environment

```bash
python setup_chess_training.py
```

This will:
- Install required dependencies (JAX, Haiku, etc.)
- Check your Python environment
- Verify data files
- Make scripts executable

### 2. Prepare Training Data

```bash
python prepare_chess_data.py --input chess_evals.jsonl --output-dir chess_data
```

This converts your JSONL evaluation data to the bag format expected by the model.

### 3. Start Training

```bash
python train_chess.py --data-path chess_data
```

## Configuration Options

### Model Sizes
- `--model-size small`: 4 layers, 4 heads, 64-dim embeddings (default)
- `--model-size medium`: 8 layers, 8 heads, 128-dim embeddings  
- `--model-size large`: 12 layers, 12 heads, 256-dim embeddings

### Training Parameters
- `--batch-size 32`: Training batch size
- `--learning-rate 1e-4`: Learning rate for Adam optimizer
- `--num-steps 10000`: Number of training steps
- `--num-return-buckets 128`: Number of value buckets for discretization

### Example Commands

```bash
# Small model for quick testing
python train_chess.py --data-path chess_data --model-size small --num-steps 5000

# Medium model for better performance
python train_chess.py --data-path chess_data --model-size medium --batch-size 64 --num-steps 20000

# Large model with full training
python train_chess.py --data-path chess_data --model-size large --batch-size 32 --num-steps 50000
```

## Data Format

The model expects your evaluation data in JSONL format:
```json
{"variant": "chess", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.524}
```

Where:
- `fen`: Board position in Forsyth-Edwards Notation
- `value`: Win probability in [0,1] range (logistic transformed)
- `variant`: Should be "chess"

## Model Architecture

The model uses:
- **Tokenization**: FEN strings → 77-token sequences
- **Transformer**: Decoder-only architecture with learned positional encodings
- **Output**: Discretized value predictions using return buckets
- **Training**: State-value prediction with cross-entropy loss

## Key Features

1. **Standard Chess**: Uses regular chess rules and evaluation
2. **FEN Tokenization**: Converts board positions to 77-token sequences
3. **State-Value Learning**: Learns position evaluations without search

## Files Created

- `prepare_chess_data.py`: Data preparation script
- `train_chess.py`: Main training script
- `chess_train_config.py`: Training configuration
- `setup_chess_training.py`: Environment setup
- `searchless_chess/`: Cloned Google DeepMind repository

## Monitoring Training

The model will log training progress every 100 steps and save checkpoints every 1000 steps. Look for:

```
Step 100: loss=2.34, accuracy=0.42
Step 200: loss=2.12, accuracy=0.46
...
✓ Checkpoint saved at step 1000
```

## Next Steps

After training, you can:
1. Evaluate the model on test positions
2. Use the trained model for position analysis
3. Compare performance against your original evaluation engine
4. Fine-tune with additional data

## Troubleshooting

- **Out of Memory**: Reduce `--batch-size`
- **Slow Training**: Install JAX with GPU support
- **Data Errors**: Check your JSONL format matches expected schema
- **Import Errors**: Run `setup_chess_training.py` again

## Requirements

- Python 3.8+
- JAX (with optional GPU support)
- Haiku, Optax, Chex
- python-chess
- Your chess evaluation data in JSONL format