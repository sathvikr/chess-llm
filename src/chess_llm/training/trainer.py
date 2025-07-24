import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from loguru import logger

from chess_llm.core.config import Config
from chess_llm.core.exceptions import TrainingError, CheckpointError
from chess_llm.data.loader import DataLoader
from chess_llm.models.transformer import create_model, initialize_model, model_summary


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.step = 0
        self.best_loss = float('inf')
        
        self.model = create_model(config.model)
        self.optimizer = self._create_optimizer()
        
        self._setup_checkpointing()

    def _create_optimizer(self) -> optax.GradientTransformation:
        if self.config.training.num_steps <= self.config.training.warmup_steps:
            schedule = optax.constant_schedule(self.config.training.learning_rate)
        else:
            decay_steps = max(1, self.config.training.num_steps - self.config.training.warmup_steps)
            warmup_steps = min(self.config.training.warmup_steps, self.config.training.num_steps - 1)
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.training.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=self.config.training.learning_rate * 0.1
            )
        
        return optax.chain(
            optax.clip_by_global_norm(self.config.training.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.training.weight_decay
            )
        )

    def _setup_checkpointing(self) -> None:
        self.checkpoint_dir = self.config.checkpoint_dir / self.config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpointer = ocp.CheckpointManager(
            directory=self.checkpoint_dir,
            checkpointers={'state': ocp.StandardCheckpointer()},
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                create=True
            )
        )

    def _loss_fn(self, params: Dict[str, Any], rng_key: jax.random.PRNGKey, tokens: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        logits = self.model.apply(params, rng_key, tokens, is_training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    def _train_step(
        self, 
        params: Dict[str, Any], 
        opt_state: optax.OptState, 
        rng_key: jax.random.PRNGKey,
        tokens: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Tuple[Dict[str, Any], optax.OptState, Dict[str, float]]:
        
        dropout_key, eval_key = jax.random.split(rng_key)
        
        def loss_fn(p):
            return self._loss_fn(p, dropout_key, tokens, targets)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        logits = self.model.apply(new_params, None, tokens, is_training=False)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'learning_rate': 0.0001,  # TODO: Fix learning rate access
        }
        
        return new_params, new_opt_state, metrics

    def _evaluate(self, params: Dict[str, Any], eval_loader: DataLoader) -> Dict[str, float]:
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for tokens, targets in eval_loader:
            logits = self.model.apply(params, None, tokens, is_training=False)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
            
            total_loss += float(jnp.mean(loss))
            total_accuracy += float(accuracy)
            num_batches += 1
        
        if num_batches == 0:
            raise TrainingError("No evaluation batches found")
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_accuracy': total_accuracy / num_batches
        }

    def save_checkpoint(self, params: Dict[str, Any], opt_state: optax.OptState, metrics: Dict[str, float]) -> None:
        try:
            state = {
                'params': params,
                'opt_state': opt_state,
                'step': self.step,
                'best_loss': self.best_loss,
                'metrics': metrics,
                'config': self.config.model_dump(),
                'timestamp': time.time()
            }
            
            self.checkpointer.save(
                step=self.step,
                items={'state': state}
            )
            
            logger.info(f"Checkpoint saved at step {self.step}")
            
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Tuple[Dict[str, Any], optax.OptState]]:
        try:
            latest_step = self.checkpointer.latest_step()
            if latest_step is None:
                logger.info("No existing checkpoint found")
                return None
            
            restored = self.checkpointer.restore(latest_step)
            state = restored['state']
            
            self.step = state['step']
            self.best_loss = state['best_loss']
            
            logger.info(f"Checkpoint loaded from step {self.step}")
            return state['params'], state['opt_state']
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        logger.info("Starting training...")
        logger.info(f"Training steps: {self.config.training.num_steps}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        
        rng_key = jax.random.PRNGKey(42)
        sample_batch = train_loader.get_single_batch()
        sample_tokens = sample_batch[0][:1]
        
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data is not None:
            params, opt_state = checkpoint_data
        else:
            params = initialize_model(self.model, rng_key, sample_tokens)
            opt_state = self.optimizer.init(params)
            logger.info(model_summary(params, self.config.model))
        
        train_step_jit = jax.jit(self._train_step)
        
        start_time = time.time()
        
        try:
            while self.step < self.config.training.num_steps:
                for tokens, targets in train_loader:
                    if self.step >= self.config.training.num_steps:
                        break
                    
                    rng_key, step_key = jax.random.split(rng_key)
                    params, opt_state, metrics = train_step_jit(params, opt_state, step_key, tokens, targets)
                    self.step += 1
                    
                    if self.step % self.config.training.log_frequency == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = self.step / elapsed
                        
                        log_msg = f"Step {self.step:6d} | Loss: {metrics['loss']:.4f} | " \
                                f"Acc: {metrics['accuracy']:.4f} | LR: {metrics['learning_rate']:.2e} | " \
                                f"Steps/sec: {steps_per_sec:.2f}"
                        logger.info(log_msg)
                    
                    if eval_loader and self.step % self.config.training.eval_frequency == 0:
                        eval_metrics = self._evaluate(params, eval_loader)
                        metrics.update(eval_metrics)
                        
                        if eval_metrics['eval_loss'] < self.best_loss:
                            self.best_loss = eval_metrics['eval_loss']
                            logger.info(f"New best eval loss: {self.best_loss:.4f}")
                    
                    if self.step % self.config.training.save_frequency == 0:
                        self.save_checkpoint(params, opt_state, metrics)
                        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            raise TrainingError(f"Training failed: {e}")
        
        final_metrics = {'final_step': self.step, 'training_time': time.time() - start_time}
        if eval_loader:
            final_eval_metrics = self._evaluate(params, eval_loader)
            final_metrics.update(final_eval_metrics)
        
        self.save_checkpoint(params, opt_state, final_metrics)
        logger.info("Training completed")
        
        return final_metrics