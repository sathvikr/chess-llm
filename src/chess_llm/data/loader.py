import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Union
from loguru import logger

import jax
import jax.numpy as jnp

from chess_llm.core.exceptions import DataError


class DataLoader:
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        if not self.data_path.exists():
            raise DataError(f"Data file not found: {self.data_path}")
            
        self.tokens, self.values = self._load_data()
        self.num_samples = len(self.tokens)
        
        if self.num_samples == 0:
            raise DataError("Empty dataset")
            
        logger.info(f"Loaded {self.num_samples:,} samples from {self.data_path}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = np.load(self.data_path)
            return data['tokens'], data['values']
        except Exception as e:
            raise DataError(f"Failed to load data from {self.data_path}: {e}")

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        indices = np.arange(self.num_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            batch_tokens = self.tokens[batch_indices]
            batch_values = self.values[batch_indices]
            
            if len(batch_tokens) < self.batch_size:
                padding_size = self.batch_size - len(batch_tokens)
                pad_indices = np.random.choice(len(batch_tokens), padding_size)
                batch_tokens = np.concatenate([batch_tokens, batch_tokens[pad_indices]])
                batch_values = np.concatenate([batch_values, batch_values[pad_indices]])

            yield jnp.array(batch_tokens), jnp.array(batch_values)

    def get_single_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        iterator = iter(self)
        return next(iterator)