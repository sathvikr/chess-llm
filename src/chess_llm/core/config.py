from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class TrainingPolicy(str, Enum):
    STATE_VALUE = "state_value"
    ACTION_VALUE = "action_value"
    BEHAVIORAL_CLONING = "behavioral_cloning"


class DataConfig(BaseModel):
    input_path: Path
    output_dir: Path = Field(default=Path("data"))
    batch_size: int = Field(default=32, ge=1, le=1024)
    shuffle: bool = True
    train_ratio: float = Field(default=0.9, ge=0.1, le=0.99)
    num_workers: int = Field(default=4, ge=0, le=16)
    seed: int = Field(default=42, ge=0)

    @validator("input_path")
    def input_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Input path does not exist: {v}")
        return v


class ModelConfig(BaseModel):
    size: ModelSize = ModelSize.SMALL
    vocab_size: int = Field(default=32, ge=1)
    embedding_dim: int = Field(default=64, ge=32)
    num_heads: int = Field(default=4, ge=1)
    num_layers: int = Field(default=4, ge=1)
    max_sequence_length: int = Field(default=77, ge=1)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    use_causal_mask: bool = True

    @classmethod
    def from_size(cls, size: ModelSize) -> "ModelConfig":
        configs = {
            ModelSize.SMALL: {
                "embedding_dim": 256,
                "num_heads": 8,
                "num_layers": 8,
                "max_sequence_length": 77
            },
            ModelSize.MEDIUM: {
                "embedding_dim": 1024,
                "num_heads": 8,
                "num_layers": 8,
                "max_sequence_length": 77
            },
            ModelSize.LARGE: {
                "embedding_dim": 1024,
                "num_heads": 8,
                "num_layers": 16,
                "max_sequence_length": 77
            },
        }
        return cls(size=size, **configs[size])


class TrainingConfig(BaseModel):
    policy: TrainingPolicy = TrainingPolicy.STATE_VALUE
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    num_steps: int = Field(default=10000, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    batch_size: int = Field(default=32, ge=1, le=2048)
    gradient_clip_norm: float = Field(default=1.0, ge=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    save_frequency: int = Field(default=1000, ge=1)
    log_frequency: int = Field(default=100, ge=1)
    eval_frequency: int = Field(default=1000, ge=1)
    num_return_buckets: int = Field(default=128, ge=1)
    distributed: bool = Field(default=False)
    num_gpus: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=64)

    @validator("warmup_steps")
    def warmup_less_than_total(cls, v: int, values: Dict[str, Any]) -> int:
        if "num_steps" in values and v >= values["num_steps"]:
            raise ValueError("warmup_steps must be less than num_steps")
        return v


class EvaluationConfig(BaseModel):
    batch_size: int = Field(default=64, ge=1, le=1024)
    num_samples: Optional[int] = Field(default=None, ge=1)
    metrics: List[str] = Field(default=["accuracy", "loss"])


class Config(BaseModel):
    project_name: str = "chess-transformer"
    experiment_name: str = "default"
    output_dir: Path = Field(default=Path("outputs"))
    checkpoint_dir: Path = Field(default=Path("checkpoints"))
    log_level: str = Field(default="INFO")
    
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    class Config:
        use_enum_values = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        import yaml
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def save_yaml(self, path: Union[str, Path]) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)