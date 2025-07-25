import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Union
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue
import threading

import jax
import jax.numpy as jnp

from chess_llm.core.exceptions import DataError


class DataLoader:
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        prefetch_batches: int = 4,
        num_workers: int = 2
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        
        if not self.data_path.exists():
            raise DataError(f"Data file not found: {self.data_path}")
            
        self.tokens, self.values = self._load_data()
        self.num_samples = len(self.tokens)
        
        if self.num_samples == 0:
            raise DataError("Empty dataset")
            
        self._batch_queue = None
        self._stop_prefetch = threading.Event()
        
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
        if self.prefetch_batches > 0:
            return self._iter_with_prefetch()
        else:
            return self._iter_sequential()
    
    def _iter_sequential(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Original sequential iteration for compatibility."""
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
    
    def _iter_with_prefetch(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Prefetch batches using background threads."""
        self._batch_queue = queue.Queue(maxsize=self.prefetch_batches)
        self._stop_prefetch.clear()
        
        # Start prefetch thread
        prefetch_thread = threading.Thread(target=self._prefetch_worker)
        prefetch_thread.daemon = True
        prefetch_thread.start()
        
        try:
            while True:
                item = self._batch_queue.get()
                if item is None:  # End of data
                    break
                yield item
                self._batch_queue.task_done()
        finally:
            self._stop_prefetch.set()
            prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self):
        """Background worker to prefetch batches."""
        try:
            for batch in self._iter_sequential():
                if self._stop_prefetch.is_set():
                    break
                self._batch_queue.put(batch)
            self._batch_queue.put(None)  # Signal end of data
        except Exception as e:
            logger.error(f"Prefetch worker error: {e}")
            self._batch_queue.put(None)

    def get_single_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Use sequential iteration for single batch to avoid thread overhead
        iterator = self._iter_sequential()
        return next(iterator)