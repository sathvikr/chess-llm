from typing import Dict, Any
import jax.numpy as jnp
from loguru import logger

from chess_llm.core.config import Config
from chess_llm.data.loader import DataLoader
from chess_llm.models.transformer import create_model
from chess_llm.utils.metrics import compute_metrics


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.model = create_model(config.model)

    def evaluate(self, params: Dict[str, Any], data_loader: DataLoader) -> Dict[str, float]:
        logger.info("Starting evaluation...")
        
        total_metrics = {}
        num_batches = 0
        
        for tokens, targets in data_loader:
            logits = self.model.apply(params, None, tokens, is_training=False)
            batch_metrics = compute_metrics(logits, targets)
            
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
        
        if num_batches == 0:
            logger.warning("No evaluation batches found")
            return {}
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        logger.info("Evaluation completed")
        for key, value in avg_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        return avg_metrics

    def evaluate_single_batch(self, params: Dict[str, Any], tokens: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
        logits = self.model.apply(params, None, tokens, is_training=False)
        return compute_metrics(logits, targets)