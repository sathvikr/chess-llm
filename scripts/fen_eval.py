#!/usr/bin/env python3
import chess
import chess.engine
import chess.pgn
import chess.variant
import json
import math
import click
import sys
from pathlib import Path

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

MATE_BASE = 32000

@click.command()
@click.option('--pgn-path', default="games.pgn", help='Path to PGN file.')
@click.option('--output-path', default="fen_evals.jsonl", help='Output JSONL file path.')
@click.option('--stockfish-path', default="./stockfish", help='Path to engine binary (Fairy-Stockfish or Stockfish).')
@click.option('--variant', type=click.Choice(['chess', 'antichess']), default='chess',
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

    if not Path(stockfish_path).exists():
        print(f"{RED}ERROR: Engine not found at {stockfish_path}{RESET}", file=sys.stderr)
        sys.exit(1)

    limit_kwargs = {}
    if depth > 0:
        limit_kwargs['depth'] = depth
    elif nodes > 0:
        limit_kwargs['nodes'] = nodes
    else:
        limit_kwargs['time'] = time

    print("Configuration:")
    print(f"  PGN: {pgn_path}")
    print(f"  Output: {output_path}")
    print(f"  Engine: {stockfish_path}")
    print(f"  Variant: {variant}")
    print(f"  Threads={threads} Hash={hash_size}MB  Limit={limit_kwargs}")
    print(f"  Resume from game #: {resume_from}")
    print(f"  Evaluate every Nth half-move: {every_n}")
    print(f"  Logistic: {logistic}  (scale={scale})  SurrogateMate={use_surrogate_mate}")
    print(f"  Raw output extra fields: {raw_output}")
    print(f"  AntichessBoard enabled: {antichess_board and variant=='antichess'}")
    print(f"  POV: {'White' if white_perspective else 'Side-to-move'}")
    print("-" * 60)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"{RED}Failed to start engine: {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    config_map = {"Threads": threads, "Hash": hash_size}

    if variant == 'antichess':
        config_map["UCI_Variant"] = "antichess"

    for k, v in config_map.items():
        try:
            engine.configure({k: v})
        except chess.engine.EngineError as e:
            print(f"{YELLOW}Warning: could not set {k}={v}: {e}{RESET}")

    if variant == 'antichess':
        if not engine_supports_variant(engine):
            print(f"{YELLOW}Warning: Engine does not appear to support 'antichess'. "
                  f"Proceeding in normal chess mode.{RESET}")
            variant_effective = "chess"
        else:
            variant_effective = "antichess"
    else:
        variant_effective = "chess"

    def score_components(pov_score: chess.engine.PovScore):
        if pov_score.is_mate():
            m = pov_score.mate()
            absm = abs(m)
            surrogate = MATE_BASE - 100 * (absm - 1)
            if m < 0:
                surrogate = -surrogate
            return None, m, surrogate
        else:
            cp = pov_score.score()
            return cp, None, cp

    def logistic_transform(x):
        x_clamped = max(min(x / scale, 40), -40)
        return 1.0 / (1.0 + math.exp(-x_clamped))

    game_count = 0
    visited_positions = 0
    written_positions = 0

    with open(pgn_path, encoding='utf-8', errors='replace') as pgn, \
         open(output_path, 'a', encoding='utf-8') as out:

        while game_count < resume_from:
            if chess.pgn.read_game(pgn) is None:
                print(f"{YELLOW}Reached EOF before resume point.{RESET}")
                break
            game_count += 1

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            headers = game.headers
            white_name = headers.get("White", "?")
            black_name = headers.get("Black", "?")
            result = headers.get("Result", "?")
            print(f"{GREEN}Game {game_count:,}{RESET}: {white_name} vs {black_name} ({result}) "
                  f"| written so far: {written_positions:,}")

            if variant_effective == 'antichess' and antichess_board:
                board = chess.variant.AntichessBoard()
            else:
                board = chess.Board()

            halfmove_index = 0
            for move in game.mainline_moves():
                try:
                    board.push(move)
                except ValueError as e:
                    print(f"{YELLOW}Skipping illegal move under board rules: {e}{RESET}")
                    continue

                halfmove_index += 1
                visited_positions += 1
                if halfmove_index % every_n != 0:
                    continue

                fen = board.fen()

                try:
                    info = engine.analyse(board, chess.engine.Limit(**limit_kwargs))
                except chess.engine.EngineTerminatedError:
                    print(f"{RED}Engine terminated unexpectedly.{RESET}")
                    break
                except chess.engine.EngineError as e:
                    print(f"{YELLOW}Engine error on analyse: {e}{RESET}")
                    continue

                if white_perspective:
                    pov = info["score"].pov(chess.WHITE)
                else:
                    pov = info["score"].pov(board.turn)

                cp, mate, surrogate = score_components(pov)

                if mate is not None and use_surrogate_mate:
                    base_numeric = surrogate
                else:
                    base_numeric = cp if cp is not None else 0

                if logistic:
                    value = logistic_transform(base_numeric)
                else:
                    value = base_numeric

                record = {
                    "variant": variant_effective,
                    "fen": fen,
                    "value": value
                }
                if raw_output:
                    record["cp"] = cp
                    record["mate"] = mate
                    record["surrogate_cp"] = surrogate
                    record["base_numeric"] = base_numeric

                out.write(json.dumps(record) + "\n")
                written_positions += 1

                if written_positions % 1000 == 0:
                    out.flush()

            game_count += 1
            if game_count % 100 == 0:
                print(f"✓ Processed {game_count:,} games | {written_positions:,} positions written")

    try:
        engine.quit()
    except Exception:
        pass

    print("\nSummary:")
    print(f"  Games processed: {game_count:,}")
    print(f"  Positions visited: {visited_positions:,}")
    print(f"  Positions written: {written_positions:,}")
    print(f"  Output file: {output_path}")


def engine_supports_variant(engine: chess.engine.SimpleEngine) -> bool:
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
