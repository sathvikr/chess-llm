#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import chess
import chess.engine
import chess.pgn
import chess.variant
import click


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


class Constants:
    MATE_BASE = 32000
    DEFAULT_LOGISTIC_SCALE = 300
    FLUSH_INTERVAL = 1000
    PROGRESS_INTERVAL = 100


@dataclass
class Config:
    """Configuration for FEN evaluation."""
    pgn_path: str = "games.pgn"
    output_path: str = "fen_evals.jsonl"
    stockfish_path: str = "/opt/homebrew/bin/fairy-stockfish"
    variant: str = "antichess"
    time: float = 0.050
    nodes: int = 0
    depth: int = 0
    threads: int = 4
    hash_size: int = 512
    scale: int = 300
    resume_from: int = 0
    every_n: int = 1
    logistic: bool = True
    use_surrogate_mate: bool = True
    raw_output: bool = False
    antichess_board: bool = True
    white_perspective: bool = False
    
    @property
    def limit_kwargs(self) -> Dict[str, Any]:
        """Get engine limit parameters."""
        if self.depth > 0:
            return {'depth': self.depth}
        elif self.nodes > 0:
            return {'nodes': self.nodes}
        else:
            return {'time': self.time}
    
    def validate(self) -> None:
        """Validate configuration."""
        if not Path(self.stockfish_path).exists():
            raise FileNotFoundError(f"Engine not found at {self.stockfish_path}")
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("Configuration:")
        print(f"  PGN: {self.pgn_path}")
        print(f"  Output: {self.output_path}")
        print(f"  Engine: {self.stockfish_path}")
        print(f"  Variant: {self.variant}")
        print(f"  Threads={self.threads} Hash={self.hash_size}MB  Limit={self.limit_kwargs}")
        print(f"  Resume from game #: {self.resume_from}")
        print(f"  Evaluate every Nth half-move: {self.every_n}")
        print(f"  Logistic: {self.logistic}  (scale={self.scale})  SurrogateMate={self.use_surrogate_mate}")
        print(f"  Raw output extra fields: {self.raw_output}")
        print(f"  AntichessBoard enabled: {self.antichess_board and self.variant=='antichess'}")
        print(f"  POV: {'White' if self.white_perspective else 'Side-to-move'}")
        print("-" * 60)

class EngineManager:
    """Manages chess engine configuration and operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine: Optional[chess.engine.SimpleEngine] = None
    
    def __enter__(self) -> chess.engine.SimpleEngine:
        """Start engine with context manager."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            self._configure_engine()
            return self.engine
        except Exception as e:
            print(f"{Colors.RED}Failed to start engine: {e}{Colors.RESET}", file=sys.stderr)
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up engine."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
    
    def _configure_engine(self) -> None:
        """Configure engine options."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        config_map = {
            "Threads": self.config.threads,
            "Hash": self.config.hash_size
        }
        
        # Only try to set UCI_Variant if the engine supports manual configuration
        if self.config.variant == 'antichess' and self._can_set_uci_variant():
            config_map["UCI_Variant"] = "antichess"
        
        for key, value in config_map.items():
            try:
                self.engine.configure({key: value})
            except chess.engine.EngineError as e:
                print(f"{Colors.YELLOW}Warning: could not set {key}={value}: {e}{Colors.RESET}")
    
    def _can_set_uci_variant(self) -> bool:
        """Check if engine allows manual UCI_Variant configuration."""
        try:
            if "UCI_Variant" not in self.engine.options:
                return False
            
            # Try to get the option info to see if it's read-only or automatically managed
            option = self.engine.options["UCI_Variant"]
            option_str = str(option).lower()
            
            # If the option mentions being automatically managed, don't try to set it
            if any(phrase in option_str for phrase in ["automatic", "managed", "read-only"]):
                return False
                
            return True
        except Exception:
            return False


class ScoreProcessor:
    """Handles chess engine score processing and transformations."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_components(self, pov_score: chess.engine.PovScore) -> Tuple[Optional[int], Optional[int], int]:
        """Extract cp, mate, and surrogate values from POV score."""
        if pov_score.is_mate():
            mate_value = pov_score.mate()
            abs_mate = abs(mate_value)
            surrogate = Constants.MATE_BASE - 100 * (abs_mate - 1)
            if mate_value < 0:
                surrogate = -surrogate
            return None, mate_value, surrogate
        else:
            cp_value = pov_score.score()
            return cp_value, None, cp_value
    
    def apply_logistic_transform(self, value: float) -> float:
        """Apply logistic transformation to a score."""
        clamped_value = max(min(value / self.config.scale, 40), -40)
        return 1.0 / (1.0 + math.exp(-clamped_value))
    
    def process_score(self, pov_score: chess.engine.PovScore) -> Tuple[float, Dict[str, Any]]:
        """Process a POV score into final value and optional raw data."""
        cp, mate, surrogate = self.extract_components(pov_score)
        
        if mate is not None and self.config.use_surrogate_mate:
            base_numeric = surrogate
        else:
            base_numeric = cp if cp is not None else 0
        
        if self.config.logistic:
            final_value = self.apply_logistic_transform(base_numeric)
        else:
            final_value = base_numeric
        
        raw_data = {}
        if self.config.raw_output:
            raw_data.update({
                "cp": cp,
                "mate": mate,
                "surrogate_cp": surrogate,
                "base_numeric": base_numeric
            })
        
        return final_value, raw_data


@click.command()
@click.option('--pgn-path', default="games.pgn", help='Path to PGN file.')
@click.option('--output-path', default="fen_evals.jsonl", help='Output JSONL file path.')
@click.option('--stockfish-path', default="/opt/homebrew/bin/fairy-stockfish", help='Path to engine binary (Fairy-Stockfish or Stockfish).')
@click.option('--variant', type=click.Choice(['chess', 'antichess']), default='antichess',
              help='Variant to evaluate.')
@click.option('--time', default=0.050, show_default=True,
              help='Time (seconds) per position if depth/nodes not given.')
@click.option('--nodes', default=0, show_default=True, help='Nodes limit (0 = unused).')
@click.option('--depth', default=0, show_default=True, help='Depth limit (0 = unused).')
@click.option('--threads', default=4, show_default=True, help='Engine Threads.')
@click.option('--hash', 'hash_size', default=512, show_default=True, help='Hash size in MB.')
@click.option('--scale', default=300, show_default=True, help='Logistic scale divisor for cp→prob.')
@click.option('--resume-from', default=0, show_default=True, help='Game index to resume from (0-based).')
@click.option('--every-n', default=1, show_default=True, help='Evaluate every Nth half-move (1=all).')
@click.option('--logistic/--no-logistic', default=True, show_default=True,
              help='Apply logistic transform to cp/surrogate.')
@click.option('--use-surrogate-mate/--no-use-surrogate-mate', default=True, show_default=True,
              help='Convert mate scores to large ±cp before logistic (if logistic).')
@click.option('--raw-output', is_flag=True, help='Include raw cp/mate/surrogate fields in JSON.')
@click.option('--antichess-board/--no-antichess-board', default=True, show_default=True,
              help='Use python-chess AntichessBoard when --variant antichess.')
@click.option('--white-perspective', is_flag=True,
              help='Always convert eval to White POV instead of side-to-move POV.')
def main(pgn_path, output_path, stockfish_path, variant, time, nodes, depth,
         threads, hash_size, scale, resume_from, every_n, logistic,
         use_surrogate_mate, raw_output, antichess_board, white_perspective):

    config = Config(
        pgn_path=pgn_path,
        output_path=output_path,
        stockfish_path=stockfish_path,
        variant=variant,
        time=time,
        nodes=nodes,
        depth=depth,
        threads=threads,
        hash_size=hash_size,
        scale=scale,
        resume_from=resume_from,
        every_n=every_n,
        logistic=logistic,
        use_surrogate_mate=use_surrogate_mate,
        raw_output=raw_output,
        antichess_board=antichess_board,
        white_perspective=white_perspective
    )
    
    try:
        config.validate()
    except FileNotFoundError as e:
        print(f"{Colors.RED}ERROR: {e}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)
    
    config.print_summary()

    evaluator = FENEvaluator(config)
    evaluator.run()


class FENEvaluator:
    """Main class for evaluating FEN positions from PGN games."""
    
    def __init__(self, config: Config):
        self.config = config
        self.score_processor = ScoreProcessor(config)
        self.game_count = 0
        self.visited_positions = 0
        self.written_positions = 0
    
    def run(self) -> None:
        """Run the FEN evaluation process."""
        with EngineManager(self.config) as engine:
            with open(self.config.pgn_path, encoding='utf-8', errors='replace') as pgn_file, \
                 open(self.config.output_path, 'a', encoding='utf-8') as output_file:
                
                self._skip_to_resume_point(pgn_file)
                self._process_games(pgn_file, output_file, engine)
        
        self._print_summary()
    
    def _skip_to_resume_point(self, pgn_file) -> None:
        """Skip games until we reach the resume point."""
        while self.game_count < self.config.resume_from:
            if chess.pgn.read_game(pgn_file) is None:
                print(f"{Colors.YELLOW}Reached EOF before resume point.{Colors.RESET}")
                break
            self.game_count += 1
    
    def _process_games(self, pgn_file, output_file, engine: chess.engine.SimpleEngine) -> None:
        """Process all games in the PGN file."""
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            self._process_single_game(game, output_file, engine)
            self.game_count += 1
            
            if self.game_count % Constants.PROGRESS_INTERVAL == 0:
                print(f"✓ Processed {self.game_count:,} games | {self.written_positions:,} positions written")
    
    def _process_single_game(self, game: chess.pgn.Game, output_file, engine: chess.engine.SimpleEngine) -> None:
        """Process a single game and extract positions."""
        headers = game.headers
        white_name = headers.get("White", "?")
        black_name = headers.get("Black", "?")
        result = headers.get("Result", "?")
        print(f"{Colors.GREEN}Game {self.game_count:,}{Colors.RESET}: {white_name} vs {black_name} ({result}) "
              f"| written so far: {self.written_positions:,}")
        
        board = self._create_board()
        halfmove_index = 0
        
        for move in game.mainline_moves():
            if not self._try_push_move(board, move):
                continue
            
            halfmove_index += 1
            self.visited_positions += 1
            
            if halfmove_index % self.config.every_n != 0:
                continue
            
            self._evaluate_and_write_position(board, output_file, engine)
    
    def _create_board(self) -> chess.Board:
        """Create appropriate board type based on variant."""
        if self.config.variant == 'antichess' and self.config.antichess_board:
            return chess.variant.AntichessBoard()
        else:
            return chess.Board()
    
    def _try_push_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Try to push a move to the board, handling errors gracefully."""
        try:
            board.push(move)
            return True
        except ValueError as e:
            print(f"{Colors.YELLOW}Skipping illegal move under board rules: {e}{Colors.RESET}")
            return False
    
    def _evaluate_and_write_position(self, board: chess.Board, output_file, engine: chess.engine.SimpleEngine) -> None:
        """Evaluate a position and write the result to file."""
        fen = board.fen()
        
        try:
            analysis_board = self._get_analysis_board(board)
            info = engine.analyse(analysis_board, chess.engine.Limit(**self.config.limit_kwargs))
        except chess.engine.EngineTerminatedError:
            print(f"{Colors.RED}Engine terminated unexpectedly.{Colors.RESET}")
            return
        except (chess.engine.EngineError, Exception) as e:
            print(f"{Colors.YELLOW}Engine error on analyse: {e}{Colors.RESET}")
            return
        
        if "score" not in info:
            print(f"{Colors.YELLOW}No score in engine response{Colors.RESET}")
            return
        
        pov_score = self._get_perspective_score(info["score"], board)
        value, raw_data = self.score_processor.process_score(pov_score)
        
        record = {
            "variant": self.config.variant,
            "fen": fen,
            "value": value
        }
        record.update(raw_data)
        
        output_file.write(json.dumps(record) + "\n")
        self.written_positions += 1
        
        if self.written_positions % Constants.FLUSH_INTERVAL == 0:
            output_file.flush()
    
    def _get_analysis_board(self, board: chess.Board) -> chess.Board:
        """Get the appropriate board for engine analysis."""
        if self.config.variant == 'antichess':
            return chess.variant.AntichessBoard(board.fen())
        else:
            return board
    
    def _get_perspective_score(self, score: chess.engine.Score, board: chess.Board) -> chess.engine.PovScore:
        """Get score from the appropriate perspective."""
        if self.config.white_perspective:
            return score.pov(chess.WHITE)
        else:
            return score.pov(board.turn)
    
    def _print_summary(self) -> None:
        """Print final summary statistics."""
        print("\nSummary:")
        print(f"  Games processed: {self.game_count:,}")
        print(f"  Positions visited: {self.visited_positions:,}")
        print(f"  Positions written: {self.written_positions:,}")
        print(f"  Output file: {self.config.output_path}")


def engine_supports_variant(engine: chess.engine.SimpleEngine) -> bool:
    """Check if engine supports antichess variant."""
    try:
        opts = engine.options
    except Exception:
        return False

    if "UCI_Variant" in opts:
        opt = engine.options["UCI_Variant"]
        choices = getattr(opt, 'choices', None)
        if choices and any(ch in choices for ch in ["antichess", "suicide", "giveaway"]):
            return True
        s = str(opt)
        if any(word in s.lower() for word in ["antichess", "giveaway", "suicide"]):
            return True
    return False


if __name__ == "__main__":
    main()
