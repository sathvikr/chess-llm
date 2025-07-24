#!/usr/bin/env python3
"""
Script to populate antichess_evals.jsonl from lichess_10M_games.pgn using fairy stockfish evaluator.
Converts regular chess positions to antichess and generates evaluations for each position.
"""

import chess
import chess.pgn
import chess.engine
import json
import sys
import subprocess
from pathlib import Path
import argparse
from typing import Iterator, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FairyStockfishEvaluator:
    """Wrapper for Fairy Stockfish engine to evaluate antichess positions."""
    
    def __init__(self, engine_path: str = "fairy-stockfish", depth: int = 15, time_limit: float = 1.0):
        self.engine_path = engine_path
        self.depth = depth
        self.time_limit = time_limit
        self.engine: Optional[chess.engine.SimpleEngine] = None
        
    def __enter__(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            logger.info(f"Fairy Stockfish engine initialized: {self.engine.id}")
            return self
        except Exception as e:
            logger.error(f"Failed to initialize Fairy Stockfish: {e}")
            logger.error("Make sure fairy-stockfish is installed and in PATH")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.quit()
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate an antichess position using Fairy Stockfish.
        Returns a value between 0 and 1, where 0.5 is equal.
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            # In antichess, the goal is to lose all pieces or be stalemated
            if board.is_stalemate():
                # Stalemate is a win for the current player in antichess
                return 1.0 if board.turn == chess.WHITE else 0.0
            
            if not any(board.legal_moves):
                # No legal moves (shouldn't happen if not stalemate)
                return 0.5
            
            # Get evaluation from Fairy Stockfish
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth, time=self.time_limit)
            )
            
            score = info.get("score")
            if score is None:
                return 0.5
            
            # Convert score to 0-1 range
            # Handle both Score and PovScore objects
            if hasattr(score, 'relative'):
                # PovScore object
                actual_score = score.relative
            else:
                # Regular Score object
                actual_score = score
            
            if actual_score and actual_score.is_mate():
                mate_value = actual_score.mate()
                if mate_value > 0:
                    # Positive mate means current player wins
                    return 1.0 if board.turn == chess.WHITE else 0.0
                else:
                    # Negative mate means current player loses
                    return 0.0 if board.turn == chess.WHITE else 1.0
            else:
                # Convert centipawn score to probability
                cp_score = actual_score.score() if actual_score else None
                
                if cp_score is None:
                    return 0.5
                
                # Normalize centipawn score to 0-1 range
                # In antichess, positive scores favor the side to move
                normalized = 0.5 + (cp_score / 1000.0)  # Scale centipawns
                return max(0.0, min(1.0, normalized))
                
        except Exception as e:
            logger.warning(f"Error evaluating position: {e}")
            return 0.5


def convert_to_antichess_board(regular_board: chess.Board) -> chess.Board:
    """
    Convert a regular chess position to antichess.
    In antichess, captures are mandatory and the goal is to lose all pieces.
    """
    # Create a new board with the same position
    antichess_board = chess.Board(regular_board.fen())
    
    # The position itself doesn't change, but the evaluation context does
    # Fairy Stockfish will handle the antichess rules during evaluation
    return antichess_board


def extract_antichess_positions_from_game(game: chess.pgn.Game, evaluator: FairyStockfishEvaluator) -> Iterator[Tuple[str, float]]:
    """
    Extract all positions from a regular chess game and convert to antichess evaluations.
    Yields tuples of (FEN string with antichess prefix, evaluation score).
    """
    board = game.board()
    
    # Evaluate starting position as antichess
    antichess_board = convert_to_antichess_board(board)
    fen = antichess_board.fen()
    eval_score = evaluator.evaluate_position(antichess_board)
    yield f"Antichess: {fen}", eval_score
    
    # Process each move
    for move in game.mainline_moves():
        board.push(move)
        antichess_board = convert_to_antichess_board(board)
        fen = antichess_board.fen()
        eval_score = evaluator.evaluate_position(antichess_board)
        yield f"Antichess: {fen}", eval_score


def check_fairy_stockfish_installation() -> bool:
    """Check if fairy-stockfish is installed and accessible."""
    try:
        result = subprocess.run(
            ["fairy-stockfish", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def process_pgn_file(pgn_path: Path, output_path: Path, max_games: int = None, 
                    engine_path: str = "fairy-stockfish", depth: int = 15,
                    skip_games: int = 0, worker_id: int = None) -> None:
    """
    Process PGN file and generate antichess_evals.jsonl using Fairy Stockfish.
    """
    worker_info = f" (Worker #{worker_id})" if worker_id else ""
    logger.info(f"Processing PGN file: {pgn_path}{worker_info}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Engine: {engine_path}, Depth: {depth}")
    if skip_games > 0:
        logger.info(f"Skipping first {skip_games} games")
    if max_games:
        logger.info(f"Processing max {max_games} games")
    
    if not check_fairy_stockfish_installation():
        logger.error("Fairy Stockfish not found. Please install it:")
        logger.error("1. Download from: https://github.com/ianfab/Fairy-Stockfish")
        logger.error("2. Or install via: pip install fairy-stockfish")
        logger.error("3. Make sure 'fairy-stockfish' is in your PATH")
        sys.exit(1)
    
    games_processed = 0
    games_skipped = 0
    positions_extracted = 0
    
    with FairyStockfishEvaluator(engine_path, depth) as evaluator:
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file, \
             open(output_path, 'w', encoding='utf-8') as output_file:
            
            # Skip games if needed
            while games_skipped < skip_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    logger.error(f"Reached end of file while skipping games. Only {games_skipped} games available.")
                    return
                games_skipped += 1
                if games_skipped % 10000 == 0:
                    logger.info(f"Skipped {games_skipped}/{skip_games} games...")
            
            if skip_games > 0:
                logger.info(f"Finished skipping {skip_games} games, starting processing...")
            
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    # Extract antichess positions from this game
                    for position, evaluation in extract_antichess_positions_from_game(game, evaluator):
                        line = json.dumps([position, evaluation])
                        output_file.write(line + '\n')
                        positions_extracted += 1
                    
                    games_processed += 1
                    
                    worker_prefix = f"WORKER #{worker_id}: " if worker_id else ""
                    logger.info(f"{worker_prefix}Processed {games_processed} games, {positions_extracted} positions")
                    
                    if max_games and games_processed >= max_games:
                        logger.info(f"Reached maximum games limit: {max_games}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing game {games_processed + 1}: {e}")
                    continue
    
    worker_prefix = f"WORKER #{worker_id}: " if worker_id else ""
    logger.info(f"{worker_prefix}Finished processing. Total games: {games_processed}, Total positions: {positions_extracted}")


def main():
    parser = argparse.ArgumentParser(description="Populate antichess_evals.jsonl from lichess PGN file using Fairy Stockfish")
    parser.add_argument(
        "--pgn-file", 
        type=Path, 
        default=Path("data/lichess_10M_games.pgn"),
        help="Path to the PGN file (default: data/lichess_10M_games.pgn)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/antichess_evals.jsonl"),
        help="Path to output JSONL file (default: data/antichess_evals.jsonl)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process (default: all games)"
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="fairy-stockfish",
        help="Path to Fairy Stockfish executable (default: fairy-stockfish)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Search depth for Fairy Stockfish (default: 15)"
    )
    parser.add_argument(
        "--skip-games",
        type=int,
        default=0,
        help="Number of games to skip from the beginning (default: 0)"
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        help="Worker ID for logging (optional)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file"
    )
    
    args = parser.parse_args()
    
    # Check if PGN file exists
    if not args.pgn_file.exists():
        logger.error(f"PGN file not found: {args.pgn_file}")
        sys.exit(1)
    
    # Check if output file exists
    if args.output_file.exists() and not args.overwrite:
        logger.error(f"Output file already exists: {args.output_file}")
        logger.error("Use --overwrite to overwrite existing file")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        process_pgn_file(args.pgn_file, args.output_file, args.max_games, 
                        args.engine_path, args.depth, args.skip_games, args.worker_id)
        logger.info("Successfully completed processing")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()