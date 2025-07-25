import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from loguru import logger

from chess_llm.core.constants import (
    CHARACTERS_INDEX, 
    SPACES_CHARACTERS, 
    SEQUENCE_LENGTH
)
from chess_llm.core.exceptions import DataError


class DataProcessor:
    def __init__(self, train_ratio: float = 0.9, seed: int = 42):
        self.train_ratio = train_ratio
        self.seed = seed
        np.random.seed(seed)

    def load_evaluations(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise DataError(f"Evaluation file not found: {path}")

        evaluations = []
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    processed = self._process_evaluation_line(data, line_num)
                    if processed:
                        evaluations.append(processed)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Data processing error at line {line_num}: {e}")
                    continue

        if not evaluations:
            raise DataError("No valid evaluations found in input file")
        
        logger.info(f"Loaded {len(evaluations):,} evaluations from {path}")
        return evaluations

    def _process_evaluation_line(self, data: Any, line_num: int) -> Dict[str, Any]:
        if isinstance(data, list) and len(data) == 2:
            fen, value = data
            if isinstance(fen, str):
                # Handle both prefixed and non-prefixed FEN strings
                if fen.startswith("Chess: "):
                    fen = fen[7:]
                return {"fen": fen, "value": float(value), "variant": "chess"}
        elif isinstance(data, dict) and 'fen' in data and 'value' in data:
            return {
                "fen": data['fen'],
                "value": float(data['value']),
                "variant": data.get('variant', 'chess')
            }
        return None

    def tokenize_fen(self, fen: str) -> np.ndarray:
        try:
            board, side, castling, en_passant, halfmoves, fullmoves = fen.split(' ')
            board = board.replace('/', '')
            board = side + board

            indices = []
            for char in board:
                if char in SPACES_CHARACTERS:
                    indices.extend(int(char) * [CHARACTERS_INDEX['.']])
                else:
                    indices.append(CHARACTERS_INDEX[char])

            if castling == '-':
                indices.extend(4 * [CHARACTERS_INDEX['.']])
            else:
                for char in castling:
                    indices.append(CHARACTERS_INDEX[char])
                if len(castling) < 4:
                    indices.extend((4 - len(castling)) * [CHARACTERS_INDEX['.']])

            if en_passant == '-':
                indices.extend(2 * [CHARACTERS_INDEX['.']])
            else:
                for char in en_passant:
                    indices.append(CHARACTERS_INDEX[char])

            halfmoves += '.' * (3 - len(halfmoves))
            indices.extend([CHARACTERS_INDEX[x] for x in halfmoves])

            fullmoves += '.' * (3 - len(fullmoves))
            indices.extend([CHARACTERS_INDEX[x] for x in fullmoves])

            if len(indices) != SEQUENCE_LENGTH:
                raise DataError(f"Invalid tokenized sequence length: {len(indices)}")

            return np.asarray(indices, dtype=np.uint8)
        except Exception as e:
            raise DataError(f"Failed to tokenize FEN '{fen}': {e}")

    def convert_to_buckets(self, value: float, num_buckets: int = 128) -> int:
        value = max(0.0, min(1.0, value))
        bucket_idx = int(value * num_buckets)
        return min(bucket_idx, num_buckets - 1)

    def create_state_value_data(
        self, evaluations: List[Dict[str, Any]], num_buckets: int = 128
    ) -> List[Tuple[np.ndarray, int]]:
        data = []
        for eval_data in evaluations:
            try:
                tokens = self.tokenize_fen(eval_data['fen'])
                bucket = self.convert_to_buckets(eval_data['value'], num_buckets)
                data.append((tokens, bucket))
            except Exception as e:
                logger.warning(f"Skipping evaluation due to error: {e}")
                continue
        return data

    def split_data(self, data: List[Any]) -> Tuple[List[Any], List[Any]]:
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        
        split_idx = int(len(shuffled_data) * self.train_ratio)
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        logger.info(f"Split data: {len(train_data):,} train, {len(test_data):,} test")
        return train_data, test_data

    def process_file(
        self, 
        input_path: Union[str, Path], 
        output_dir: Union[str, Path],
        num_buckets: int = 128
    ) -> Dict[str, int]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.npz"
        test_path = output_dir / "test.npz"

        train_tokens, train_values = [], []
        test_tokens, test_values = [], []

        with open(input_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    processed = self._process_evaluation_line(data, 0)
                    if processed:
                        tokens = self.tokenize_fen(processed['fen'])
                        bucket = self.convert_to_buckets(processed['value'], num_buckets)
                        
                        if np.random.rand() < self.train_ratio:
                            train_tokens.append(tokens)
                            train_values.append(bucket)
                        else:
                            test_tokens.append(tokens)
                            test_values.append(bucket)
                except (json.JSONDecodeError, DataError) as e:
                    logger.warning(f"Skipping line due to error: {e}")
                    continue

        self._save_numpy_data_from_lists(train_tokens, train_values, train_path)
        self._save_numpy_data_from_lists(test_tokens, test_values, test_path)

        return {
            "total": len(train_tokens) + len(test_tokens),
            "train": len(train_tokens),
            "test": len(test_tokens)
        }

    def _save_numpy_data_from_lists(self, tokens: List[np.ndarray], values: List[int], path: Path) -> None:
        if not tokens:
            logger.warning(f"No data to save to {path}")
            return

        tokens_array = np.stack(tokens)
        values_array = np.array(values)

        np.savez_compressed(path, tokens=tokens_array, values=values_array)
        logger.info(f"Saved {len(tokens)} samples to {path}")

    def _save_numpy_data(self, data: List[Tuple[np.ndarray, int]], path: Path) -> None:
        if not data:
            raise DataError("Cannot save empty dataset")
            
        tokens = np.stack([item[0] for item in data])
        values = np.array([item[1] for item in data])
        
        np.savez_compressed(path, tokens=tokens, values=values)
        logger.info(f"Saved {len(data):,} samples to {path}")

    def load_numpy_data(self, path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(path)
        if not path.exists():
            raise DataError(f"Data file not found: {path}")
            
        data = np.load(path)
        return data['tokens'], data['values']