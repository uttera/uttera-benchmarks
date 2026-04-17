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
```

## Observations

- 1884/1884 requests succeeded across all six profiles.
- `route` is `-` for every request: the vLLM wrapper does not set an `X-Route` header (there is nothing to route — the engine handles every request uniformly via continuous batching).
- RPS saturates around 18 from N=64 upwards. Below that the per-request pipeline overhead dominates.
