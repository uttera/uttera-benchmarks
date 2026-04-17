# Run 1 — hotcold, GPU shared-but-idle with vLLM

**Date:** 2026-04-17
**Target:** `sphinx:5000` (`uttera-stt-hotcold`, `openai-whisper` Whisper-large-v3-turbo)
**Corpus:** `librispeech-test-clean` (2620 FLAC clips, 4-20 s, 16 kHz mono)
**Harness:** `bench.py` at commit _(set after this file is committed)_

## Setup

- Host: `sphinx`, shared RTX 5090 (32 GB) with `openclaw` LXC via GPU pass-through (same GPU UUID, confirmed with `nvidia-smi --query-gpu=uuid`).
- Before Run 1: the local `uttera-stt-vllm` on `openclaw:5002` was **killed** to free its 22 GB VRAM reservation. Without this step the hotcold cold-pool could not spawn because VRAM was starved.
- `sphinx:5100` (TTS) was alive but idle — no traffic during the run. Working hypothesis is that idle neighbours don't eat GPU compute. Run 2 will test this.
- No requests from outside the benchmark harness hit `sphinx:5000` during the run.

## Commands

```bash
./bench.py --mode stt --server http://sphinx:5000 --profile latency            --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-latency.json
./bench.py --mode stt --server http://sphinx:5000 --profile burst --n 8        --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst8.json
./bench.py --mode stt --server http://sphinx:5000 --profile burst --n 64       --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst64.json
./bench.py --mode stt --server http://sphinx:5000 --profile burst --n 256      --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst256.json
./bench.py --mode stt --server http://sphinx:5000 --profile burst --n 512      --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst512.json
./bench.py --mode stt --server http://sphinx:5000 --profile burst --n 1024     --corpus ./corpora/librispeech-test-clean --output results/.../hotcold-burst1024.json
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

## Anomalies / notes

- All 1884 requests succeeded (OK=N, fail=0 in every profile).
- `route` distribution crosses from "all HOT" (up to N=64) to "HOT + COLD-POOL" (N=256+). The cold pool spawns workers on demand; with the full GPU available, it reached ~6 cold workers under N=1024.
- No cache-like effect observed — latency scales monotonically with N, and route distribution is consistent across runs. The earlier "cache confusion" (see `README.md > What we got wrong the first time`) was an artefact of a 160-clip corpus with repetitions, not a real server-side cache.

## Open questions

- What is hotcold's exact `COLD_POOL_SIZE` in this deployment? The route histogram tops out at ~545 COLD-POOL at N=1024, suggesting 6–8 pool workers in rotation.
- Tail latency at N≥512 is very long (p95 ~50–100 s). Is this acceptable for Uttera's target traffic pattern? That's a business question, not a benchmarking one.
