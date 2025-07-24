import jax.numpy as jnp
import optax
from typing import Dict


def calculate_accuracy(logits: jnp.ndarray, targets: jnp.ndarray) -> float:
    predictions = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(predictions == targets))


def calculate_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> float:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return float(jnp.mean(loss))


def calculate_top_k_accuracy(logits: jnp.ndarray, targets: jnp.ndarray, k: int = 5) -> float:
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
    targets_expanded = jnp.expand_dims(targets, axis=-1)
    matches = jnp.any(top_k_preds == targets_expanded, axis=-1)
    return float(jnp.mean(matches))


def calculate_perplexity(logits: jnp.ndarray, targets: jnp.ndarray) -> float:
    loss = calculate_loss(logits, targets)
    return float(jnp.exp(loss))


def compute_metrics(logits: jnp.ndarray, targets: jnp.ndarray, prefix: str = "") -> Dict[str, float]:
    metrics = {
        f"{prefix}accuracy": calculate_accuracy(logits, targets),
        f"{prefix}loss": calculate_loss(logits, targets),
        f"{prefix}top_5_accuracy": calculate_top_k_accuracy(logits, targets, k=5),
        f"{prefix}perplexity": calculate_perplexity(logits, targets),
    }
    return metrics