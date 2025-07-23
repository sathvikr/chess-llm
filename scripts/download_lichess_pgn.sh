#!/usr/bin/env bash
# download_lichess_10M.sh
# macOS‑ready: streams Lichess standard‑rated PGNs (.zst) into one file, stops at 10 M games.

set -euo pipefail
IFS=$'\n\t'

### ─── Configuration ──────────────────────────────────────────────────────────

TARGET_GAMES=10000000
OUTPUT_PGN="lichess_10M_games.pgn"
START_YEAR=2025
START_MONTH=06

### ─── Main ───────────────────────────────────────────────────────────────────

# Create or truncate output
> "$OUTPUT_PGN"

games_collected=0
year=$START_YEAR
month=$START_MONTH

while (( games_collected < TARGET_GAMES )); do
  m=$(printf "%02d" "$month")
  url="https://database.lichess.org/standard/lichess_db_standard_rated_${year}-${m}.pgn.zst"
  echo "→ Trying $url …"

  # Attempt to download & stream; if curl fails (e.g. 404), break out
  if ! curl -fSL "$url" \
       | unzstd -c \
       | awk -v count="$games_collected" -v target="$TARGET_GAMES" '
           /^\[Event / { count++ }
           { print }
           count >= target { exit }
         ' \
       >> "$OUTPUT_PGN"; then
    echo "⚠️  No more dumps at $url; stopping."
    break
  fi

  # Update count
  games_collected=$(grep -c '^\[Event ' "$OUTPUT_PGN")
  echo "   • Total games so far: $games_collected"

  # Step month backwards
  (( month-- ))
  if (( month == 0 )); then
    month=12
    (( year-- ))
  fi
done

echo
if (( games_collected >= TARGET_GAMES )); then
  echo "✅ Done — collected $games_collected games in $OUTPUT_PGN"
else
  echo "⚠️  Finished with $games_collected games (ran out of monthly dumps)."
fi
