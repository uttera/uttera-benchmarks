# Run 1 — hotcold on LibriSpeech

**Date:** 2026-04-17
**Target:** `stt-node:5000` (`uttera-stt-hotcold`, `openai-whisper` Whisper-large-v3-turbo)
**Corpus:** `librispeech-test-clean` (2620 FLAC clips, 4–20 s, 16 kHz mono)
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8

## Setup

Only this backend was active on the GPU during the run. No other inference workload was running, no traffic hit the service from outside the bench harness, and the GPU was otherwise idle.

## Commands

```bash
./bench.py --mode stt --server http://stt-node:5000 --profile latency       --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-latency.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 8   --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst8.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 64  --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst64.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 256 --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst256.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 512 --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst512.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 1024 --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst1024.json
```

## Raw files in this folder

| File | Content |
|---|---|
| `hotcold-latency.{json,csv}` | 20 sequential requests + 3 warmup |
| `hotcold-burst8.{json,csv}` | 8 simultaneous requests |
| `hotcold-burst64.{json,csv}` | 64 simultaneous requests |
| `hotcold-burst256.{json,csv}` | 256 simultaneous requests |
| `hotcold-burst512.{json,csv}` | 512 simultaneous requests |
| `hotcold-burst1024.{json,csv}` | 1024 simultaneous requests |

Each CSV has one row per request: `clip_id, audio_seconds, latency_ms, status, route, bytes, error`. Anyone can recompute p50/p95/p99 from the raw data.

## Observations

- 1884/1884 requests succeeded across all six profiles.
- `route` distribution transitions from "all HOT" (up to N=64) to "HOT + COLD-POOL" at N=256 and above. The hotcold architecture's COLD-POOL spawns subprocess workers on demand up to its configured cap; with the full GPU available, it reached roughly 6 cold workers under N=1024.

## Open questions

- Tail latency at N≥512 is very long (p95 ~50–100 s). Whether that is acceptable depends on the target traffic pattern — it is a deployment decision, not a benchmarking one.
