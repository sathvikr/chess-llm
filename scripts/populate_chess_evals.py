#!/usr/bin/env python3
"""
Script to populate chess_evals.jsonl from lichess_10M_games.pgn using python-chess.
Extracts positions from PGN games and generates evaluations for each position.
"""

import chess
import chess.pgn
import json
import sys
from pathlib import Path
import argparse
from typing import Iterator, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_position_evaluation(board: chess.Board) -> float:
    """
    Simple position evaluation based on material balance and basic positional factors.
    Returns a value between 0 and 1, where 0.5 is equal, >0.5 favors white, <0.5 favors black.
    """
    if board.is_checkmate():
        return 0.0 if board.turn else 1.0
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    
    # Material values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = 0
    black_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    # Basic positional factors
    white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    board.turn = not board.turn
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.turn = not board.turn
    
    # Normalize material difference
    total_material = white_material + black_material
    if total_material == 0:
        material_eval = 0
    else:
        material_eval = (white_material - black_material) / total_material
    
    # Normalize mobility difference
    total_mobility = white_mobility + black_mobility
    if total_mobility == 0:
        mobility_eval = 0
    else:
        mobility_eval = (white_mobility - black_mobility) / total_mobility * 0.1
    
    # Combine evaluations
    eval_score = 0.5 + (material_eval * 0.8) + (mobility_eval * 0.2)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, eval_score))


def extract_positions_from_game(game: chess.pgn.Game) -> Iterator[Tuple[str, float]]:
    """
    Extract all positions from a game and generate evaluations.
    Yields tuples of (FEN string, evaluation score).
    """
    board = game.board()
    
    # Evaluate starting position
    fen = board.fen()
    eval_score = simple_position_evaluation(board)
    yield f"Chess: {fen}", eval_score
    
    # Process each move
    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()
        eval_score = simple_position_evaluation(board)
        yield f"Chess: {fen}", eval_score


def process_pgn_file(pgn_path: Path, output_path: Path, max_games: int = None) -> None:
    """
    Process PGN file and generate chess_evals.jsonl.
    """
    logger.info(f"Processing PGN file: {pgn_path}")
    logger.info(f"Output file: {output_path}")
    
    games_processed = 0
    positions_extracted = 0
    
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file, \
         open(output_path, 'w', encoding='utf-8') as output_file:
        
        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                # Extract positions from this game
                for position, evaluation in extract_positions_from_game(game):
                    line = json.dumps([position, evaluation])
                    output_file.write(line + '\n')
                    positions_extracted += 1
                
                games_processed += 1
                
                if games_processed % 100 == 0:
                    logger.info(f"Processed {games_processed} games, {positions_extracted} positions")
                
                if max_games and games_processed >= max_games:
                    logger.info(f"Reached maximum games limit: {max_games}")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing game {games_processed + 1}: {e}")
                continue
    
    logger.info(f"Finished processing. Total games: {games_processed}, Total positions: {positions_extracted}")


def main():
    parser = argparse.ArgumentParser(description="Populate chess_evals.jsonl from lichess PGN file")
    parser.add_argument(
        "--pgn-file", 
        type=Path, 
        default=Path("data/lichess_10M_games.pgn"),
        help="Path to the PGN file (default: data/lichess_10M_games.pgn)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/chess_evals.jsonl"),
        help="Path to output JSONL file (default: data/chess_evals.jsonl)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process (default: all games)"
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
        process_pgn_file(args.pgn_file, args.output_file, args.max_games)
        logger.info("Successfully completed processing")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()