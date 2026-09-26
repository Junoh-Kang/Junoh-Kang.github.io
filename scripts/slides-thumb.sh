#!/usr/bin/env bash
# Make blog/<post>/slides-thumb.png from the deck named by `slides:` in that post's front matter.
# PDF decks: first page via Ghostscript. HTML decks: first screen via headless Chrome.
# Usage: scripts/slides-thumb.sh blog/<post>/ [...]
set -euo pipefail
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for dir in "$@"; do
  dir="${dir%/}"
  deck="$(sed -n 's/^slides:[[:space:]]*//p' "$dir/index.md" | head -1 | tr -d "'\"")"
  [ -n "$deck" ] || { echo "$dir: no slides: in front matter" >&2; exit 1; }
  case "$deck" in
    *.pdf)
      gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r100 -dFirstPage=1 -dLastPage=1 \
        -sOutputFile="$tmp/raw.png" "$dir/$deck" ;;
    *.html)
      "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,720 \
        --virtual-time-budget=5000 --screenshot="$tmp/raw.png" "file://$PWD/$dir/$deck" 2>/dev/null ;;
    *) echo "$dir: unsupported deck type: $deck" >&2; exit 1 ;;
  esac
  magick "$tmp/raw.png" -resize 480x -strip "$dir/slides-thumb.png"
  echo "wrote $dir/slides-thumb.png"
done
