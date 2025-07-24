import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import jax
from loguru import logger

from chess_llm.core.exceptions import CheckpointError


def save_checkpoint(
    params: Dict[str, Any],
    opt_state: Any,
    step: int,
    checkpoint_dir: Path,
    prefix: str = "checkpoint"
) -> Path:
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{prefix}_step_{step}.pkl"
        
        checkpoint_data = {
            'params': params,
            'opt_state': opt_state,
            'step': step,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        raise CheckpointError(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
        
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint: {e}")


def find_latest_checkpoint(checkpoint_dir: Path, prefix: str = "checkpoint") -> Optional[Path]:
    try:
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob(f"{prefix}_step_*.pkl"))
        if not checkpoints:
            return None
        
        def extract_step(path: Path) -> int:
            stem = path.stem
            return int(stem.split('_')[-1])
        
        latest_checkpoint = max(checkpoints, key=extract_step)
        return latest_checkpoint
        
    except Exception as e:
        logger.error(f"Error finding latest checkpoint: {e}")
        return None


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 5, prefix: str = "checkpoint") -> None:
    try:
        if not checkpoint_dir.exists():
            return
        
        checkpoints = list(checkpoint_dir.glob(f"{prefix}_step_*.pkl"))
        if len(checkpoints) <= keep_last:
            return
        
        def extract_step(path: Path) -> int:
            stem = path.stem
            return int(stem.split('_')[-1])
        
        sorted_checkpoints = sorted(checkpoints, key=extract_step, reverse=True)
        checkpoints_to_remove = sorted_checkpoints[keep_last:]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint}")
            
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {e}")