from typing import Optional
import jax
import jax.numpy as jnp
import haiku as hk
from loguru import logger

from chess_llm.core.config import ModelConfig
from chess_llm.core.exceptions import ModelError


class MultiHeadAttention(hk.Module):
    def __init__(self, num_heads: int, key_size: Optional[int] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        return hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(1.0)
        )(query, key, value, mask)


class FeedForward(hk.Module):
    def __init__(self, hidden_size: int, output_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.hidden_size, w_init=hk.initializers.VarianceScaling(1.0))(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self.output_size, w_init=hk.initializers.VarianceScaling(1.0))(x)
        return x


class TransformerBlock(hk.Module):
    def __init__(self, config: ModelConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, is_training: bool = True) -> jnp.ndarray:
        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_size=self.config.embedding_dim // self.config.num_heads
        )(x, x, x, mask)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.config.dropout_rate, x)
        x = x + residual

        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = FeedForward(hidden_size=self.config.embedding_dim * 4, output_size=self.config.embedding_dim)(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.config.dropout_rate, x)
        x = x + residual

        return x


class ChessLLM(hk.Module):
    def __init__(self, config: ModelConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        
        if config.embedding_dim % config.num_heads != 0:
            raise ModelError(f"embedding_dim ({config.embedding_dim}) must be divisible by num_heads ({config.num_heads})")

    def __call__(self, tokens: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        batch_size, seq_len = tokens.shape
        
        if seq_len > self.config.max_sequence_length:
            raise ModelError(f"Sequence length {seq_len} exceeds maximum {self.config.max_sequence_length}")

        x = hk.Embed(vocab_size=self.config.vocab_size, embed_dim=self.config.embedding_dim, 
                     w_init=hk.initializers.VarianceScaling(1.0))(tokens)
        
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = hk.Embed(vocab_size=self.config.max_sequence_length, embed_dim=self.config.embedding_dim,
                          w_init=hk.initializers.VarianceScaling(1.0))(positions)
        x = x + pos_emb

        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.config.dropout_rate, x)

        mask = None
        if self.config.use_causal_mask:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = mask[None, None, :, :]

        for i in range(self.config.num_layers):
            x = TransformerBlock(self.config, name=f"layer_{i}")(x, mask, is_training)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        
        x = x[:, -1, :]
        logits = hk.Linear(128, w_init=hk.initializers.VarianceScaling(1.0))(x)
        
        return logits


def create_model(config: ModelConfig):
    def forward(tokens: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        model = ChessLLM(config)
        return model(tokens, is_training)
    
    return hk.transform(forward)


def initialize_model(model, rng_key: jax.random.PRNGKey, sample_input: jnp.ndarray):
    try:
        params = model.init(rng_key, sample_input, is_training=True)
        logger.info("Model initialized successfully")
        return params
    except Exception as e:
        raise ModelError(f"Failed to initialize model: {e}")


def model_summary(params, config: ModelConfig) -> str:
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    summary = f"""
Model Configuration:
  Architecture: Transformer
  Embedding dimension: {config.embedding_dim}
  Number of heads: {config.num_heads}
  Number of layers: {config.num_layers}
  Max sequence length: {config.max_sequence_length}
  Vocabulary size: {config.vocab_size}
  Dropout rate: {config.dropout_rate}
  Total parameters: {total_params:,}
"""
    return summary.strip()