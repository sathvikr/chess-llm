import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from chess_llm.core.config import Config, ModelSize, TrainingPolicy, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
from chess_llm.core.exceptions import ChessLLMError
from chess_llm.data.processor import DataProcessor
from chess_llm.data.loader import DataLoader
from chess_llm.training.trainer import Trainer

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def cli(log_level: str) -> None:
    setup_logging(log_level)


@cli.command()
@click.option("--input-path", "-i", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", default="data", type=click.Path(path_type=Path))
@click.option("--train-ratio", default=0.9, type=float, help="Training data ratio")
@click.option("--num-buckets", default=128, type=int, help="Number of value buckets")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num-workers", default=None, type=int, help="Number of parallel workers (default: min(16, CPU count))")
def prepare_data(
    input_path: Path,
    output_dir: Path,
    train_ratio: float,
    num_buckets: int,
    seed: int,
    num_workers: Optional[int]
) -> None:
    try:
        console.print(f"[bold blue]Preparing data from {input_path}[/bold blue]")
        
        processor = DataProcessor(train_ratio=train_ratio, seed=seed)
        stats = processor.process_file(input_path, output_dir, num_buckets, num_workers)
        
        table = Table(title="Data Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Total samples", f"{stats['total']:,}")
        table.add_row("Training samples", f"{stats['train']:,}")
        table.add_row("Test samples", f"{stats['test']:,}")
        
        console.print(table)
        console.print(f"[bold green]Data prepared successfully in {output_dir}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error preparing data: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option("--config-path", "-c", type=click.Path(path_type=Path), help="Path to config YAML file")
@click.option("--data-dir", "-d", default="data", type=click.Path(path_type=Path))
@click.option("--model-size", default="small", type=click.Choice(["small", "medium", "large"]))
@click.option("--batch-size", default=32, type=int)
@click.option("--learning-rate", default=1e-4, type=float)
@click.option("--num-steps", default=10000, type=int)
@click.option("--experiment-name", default="default", type=str)
@click.option("--no-eval", is_flag=True, help="Skip evaluation during training")
@click.pass_context
def train(
    ctx: click.Context,
    config_path: Optional[Path],
    data_dir: Path,
    model_size: str,
    batch_size: int,
    learning_rate: float,
    num_steps: int,
    experiment_name: str,
    no_eval: bool
) -> None:
    try:
        if config_path and config_path.exists():
            config = Config.from_yaml(config_path)
            console.print(f"[bold blue]Loaded config from {config_path}[/bold blue]")
        else:
            train_data_path = data_dir / "train.npz"
            test_data_path = data_dir / "test.npz"
            
            if not train_data_path.exists():
                console.print(f"[bold red]Training data not found at {train_data_path}[/bold red]")
                console.print("Run 'chess-llm prepare-data' first")
                sys.exit(1)
            
            config = Config(
                experiment_name=experiment_name,
                data=DataConfig(
                    input_path=train_data_path,
                    batch_size=batch_size
                ),
                model=ModelConfig.from_size(ModelSize(model_size)),
                training=TrainingConfig(
                    learning_rate=learning_rate,
                    num_steps=num_steps,
                    batch_size=batch_size
                ),
                evaluation=EvaluationConfig(batch_size=batch_size)
            )
        
        console.print(f"[bold blue]Starting training experiment: {config.experiment_name}[/bold blue]")

        # Prepare data before training
        ctx.invoke(
            prepare_data,
            input_path=config.data.input_path,
            output_dir=Path(config.data.output_dir),
            train_ratio=config.data.train_ratio,
            num_buckets=config.training.num_return_buckets,
            seed=config.data.seed,
            num_workers=config.data.num_workers,
        )
        
        train_loader = DataLoader(
            data_path=Path(config.data.output_dir) / "train.npz",
            batch_size=config.training.batch_size,
            shuffle=True,
            prefetch_batches=4,
            num_workers=min(4, config.data.num_workers)
        )
        
        eval_loader = None
        if not no_eval and (Path(config.data.output_dir) / "test.npz").exists():
            eval_loader = DataLoader(
                data_path=Path(config.data.output_dir) / "test.npz",
                batch_size=config.evaluation.batch_size,
                shuffle=False,
                prefetch_batches=2,
                num_workers=2
            )
        
        trainer = Trainer(config)
        results = trainer.train(train_loader, eval_loader)
        
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in results.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        console.print(f"[bold green]Training completed! Experiment: {config.experiment_name}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option("--model-size", default="small", type=click.Choice(["small", "medium", "large"]))
@click.option("--experiment-name", default="default", type=str)
@click.option("--output-path", "-o", default="config.yaml", type=click.Path(path_type=Path))
def generate_config(model_size: str, experiment_name: str, output_path: Path) -> None:
    try:
        config = Config(
            experiment_name=experiment_name,
            data=DataConfig(
                input_path=Path("data/train.npz"),
                batch_size=32
            ),
            model=ModelConfig.from_size(ModelSize(model_size)),
            training=TrainingConfig(),
            evaluation=EvaluationConfig()
        )
        
        config.save_yaml(output_path)
        console.print(f"[bold green]Configuration saved to {output_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Failed to generate config: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option("--data-dir", "-d", default="data", type=click.Path(path_type=Path))
def info(data_dir: Path) -> None:
    try:
        train_path = data_dir / "train.npz"
        test_path = data_dir / "test.npz"
        
        table = Table(title="Dataset Information")
        table.add_column("Split", style="cyan")
        table.add_column("Path", style="white")
        table.add_column("Exists", style="green")
        table.add_column("Samples", style="yellow")
        
        for name, path in [("Train", train_path), ("Test", test_path)]:
            exists = "✓" if path.exists() else "✗"
            samples = "N/A"
            
            if path.exists():
                try:
                    import numpy as np
                    data = np.load(path)
                    samples = f"{len(data['tokens']):,}"
                except Exception:
                    samples = "Error"
            
            table.add_row(name, str(path), exists, samples)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error getting info: {e}[/bold red]")
        sys.exit(1)


def main() -> None:
    try:
        cli()
    except ChessLLMError as e:
        console.print(f"[bold red]Chess LLM Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()