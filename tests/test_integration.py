import tempfile
from pathlib import Path
import numpy as np
import pytest

from chess_transformer.core.config import Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from chess_transformer.data.processor import DataProcessor
from chess_transformer.data.loader import DataLoader
from chess_transformer.training.trainer import Trainer


def create_test_data(path: Path, num_samples: int = 100):
    data = []
    for i in range(num_samples):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        value = 0.5 + 0.1 * (i % 10 - 5) / 5
        data.append(f'{{"fen": "{fen}", "value": {value}, "variant": "chess"}}')
    
    with open(path, 'w') as f:
        f.write('\n'.join(data))


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        test_jsonl = temp_path / "test_data.jsonl"
        data_dir = temp_path / "data"
        output_dir = temp_path / "outputs"
        checkpoint_dir = temp_path / "checkpoints"
        
        create_test_data(test_jsonl, 50)
        
        processor = DataProcessor(train_ratio=0.8, seed=42)
        stats = processor.process_file(test_jsonl, data_dir)
        
        assert stats['total'] == 50
        assert stats['train'] == 40
        assert stats['test'] == 10
        
        config = Config(
            experiment_name="test",
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            data=DataConfig(
                input_path=data_dir / "train.npz",
                batch_size=8
            ),
            model=ModelConfig(
                embedding_dim=32,
                num_heads=2,
                num_layers=2
            ),
            training=TrainingConfig(
                num_steps=10,
                batch_size=8,
                save_frequency=5,
                log_frequency=2
            ),
            evaluation=EvaluationConfig(batch_size=8)
        )
        
        train_loader = DataLoader(
            data_path=data_dir / "train.npz",
            batch_size=config.training.batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            data_path=data_dir / "test.npz",
            batch_size=config.evaluation.batch_size,
            shuffle=False
        )
        
        trainer = Trainer(config)
        results = trainer.train(train_loader, test_loader)
        
        assert 'final_step' in results
        assert results['final_step'] == 10
        assert 'training_time' in results
        assert 'eval_loss' in results
        assert 'eval_accuracy' in results


if __name__ == "__main__":
    test_full_pipeline()
    print("Integration test passed!")