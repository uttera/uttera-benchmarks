#!/bin/bash
# Fetch LibriSpeech test-clean, flatten into corpora/librispeech-test-clean/.
# ~350 MB download. Idempotent — skips if files already present.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DEST="$REPO_ROOT/corpora/librispeech-test-clean"

if [ -d "$DEST" ] && [ "$(ls -1 "$DEST"/*.flac 2>/dev/null | wc -l)" -eq 2620 ]; then
    echo "[*] $DEST already has 2620 FLAC files, nothing to do."
    exit 0
fi

WORK="$(mktemp -d)"
trap "rm -rf $WORK" EXIT
cd "$WORK"

echo "[*] Downloading test-clean.tar.gz (~350 MB)..."
curl -L --fail -o test-clean.tar.gz http://www.openslr.org/resources/12/test-clean.tar.gz

echo "[*] Extracting..."
tar xzf test-clean.tar.gz

echo "[*] Flattening 2620 clips into $DEST ..."
mkdir -p "$DEST"
find LibriSpeech/test-clean -name "*.flac" -exec cp {} "$DEST"/ \;

echo "[*] Done: $(ls -1 "$DEST"/*.flac | wc -l) FLAC files."
