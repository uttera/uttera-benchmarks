# Run 2 — vLLM on LibriSpeech

**Date:** 2026-04-17
**Target:** `stt-node:5000` (`uttera-stt-vllm` v0.1.0, vLLM 0.19.0, Whisper-large-v3-turbo)
**Corpus:** `librispeech-test-clean` (2620 FLAC clips, 4–20 s, 16 kHz mono)
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8

## Setup

Only this backend was active on the GPU. vLLM was started with the full production configuration:

- `gpu_memory_utilization = 0.9`
- `max_num_seqs = 64`
- `dtype = float16`
- `max_model_len = 448`

`vram_free_gb_at_start` reported by vLLM at startup: **0.87 GB** — vLLM reserved essentially the full GPU for its KV cache.

## Commands

```bash
./bench.py --mode stt --server http://stt-node:5000 --profile latency       --corpus ./corpora/librispeech-test-clean --output results/.../vllm-latency.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 8   --corpus ./corpora/librispeech-test-clean --output results/.../vllm-burst8.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 64  --corpus ./corpora/librispeech-test-clean --output results/.../vllm-burst64.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 256 --corpus ./corpora/librispeech-test-clean --output results/.../vllm-burst256.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 512 --corpus ./corpora/librispeech-test-clean --output results/.../vllm-burst512.json
./bench.py --mode stt --server http://stt-node:5000 --profile burst --n 1024 --corpus ./corpora/librispeech-test-clean --output results/.../vllm-burst1024.json
./bench.py --mode stt --server http://stt-node:5000 --profile sustained --rps 8 --duration 300 --corpus ./corpora/librispeech-test-clean --output results/.../vllm-sustained.json
```

`--rps 8` is the protocol-mandated `0.5 × rps_burst@64` (burst@64 measured 16.69 rps, so sustained runs at 50 % of peak batch capacity).

## Observations

- 4284/4284 requests succeeded across all seven profiles (1884 latency+burst + 2400 sustained, 0 failures).
- `route` is `-` for every request: the vLLM wrapper does not set an `X-Route` header (there is nothing to route — the engine handles every request uniformly via continuous batching).
- RPS saturates around 18 from N=64 upwards. Below that the per-request pipeline overhead dominates.
- **Sustained @ 8 rps / 5 min**: 2400/2400 succeeded, p50 85 ms / p95 110 ms / p99 126 ms. The p95-per-minute series is `110, 108, 114, 105, 113` ms — essentially flat, well within measurement noise. No drift, no errors, no VRAM growth. vLLM's continuous batching keeps latency stable from minute 0 (no spin-up transient beyond the 3-request warm-up).
