# Run 2a — vLLM, GPU shared-but-idle with hotcold + TTS neighbours

**Date:** 2026-04-17
**Target:** `openclaw:5002` (`uttera-stt-vllm` v0.1.0, vLLM 0.19.0, Whisper-large-v3-turbo)
**Corpus:** `librispeech-test-clean` (2620 FLAC clips, 4-20 s, 16 kHz mono)

## Setup

- Host: `openclaw` LXC container on `sphinx`, shared RTX 5090 (32 GB, Blackwell, CUDA 12.8) via GPU pass-through.
- **`sphinx:5000` (whisper-stt.service, `uttera-stt-hotcold`) was alive but idle** — no traffic.
- **`sphinx:5100` (coqui-tts.service, TTS prod) was alive but idle** — no traffic.
- vLLM arrived with heavily handicapped config because the idle neighbours held ~5 GB of VRAM: `gpu_memory_utilization=0.7, max_num_seqs=32`.
- `vram_free_gb_at_start` reported by vLLM health at startup: **2.28 GB**.

## What we set out to test

User hypothesis: **idle neighbours do not steal compute, they only hold VRAM**. If true, Run 2a (handicapped vLLM + idle neighbours) and Run 2b (vLLM with the full GPU, neighbours stopped) should produce equivalent numbers modulo noise.

## Commands

```bash
./bench.py --mode stt --server http://127.0.0.1:5002 --profile latency            --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-latency.json
./bench.py --mode stt --server http://127.0.0.1:5002 --profile burst --n 8        --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-burst8.json
./bench.py --mode stt --server http://127.0.0.1:5002 --profile burst --n 64       --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-burst64.json
./bench.py --mode stt --server http://127.0.0.1:5002 --profile burst --n 256      --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-burst256.json
./bench.py --mode stt --server http://127.0.0.1:5002 --profile burst --n 512      --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-burst512.json
./bench.py --mode stt --server http://127.0.0.1:5002 --profile burst --n 1024     --corpus ./corpora/librispeech-test-clean --shared-gpu --output results/.../vllm-burst1024.json
```

## Raw files

Six `vllm-*.{json,csv}` pairs. Each CSV: one row per request — `clip_id, audio_seconds, latency_ms, status, route, bytes, error`. `route` is `-` for vLLM (the wrapper does not set an `X-Route` header); the handler is uniform continuous batching.

## Anomalies

None. 1884/1884 requests OK across the six profiles.
