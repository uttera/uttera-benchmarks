# Run 6 — tts-vllm on `uttera-tts-40w`

**Date:** 2026-04-17
**Target:** local `uttera-tts-vllm` v0.1.2 (nano-vLLM + VoxCPM2), on `http://127.0.0.1:5100`.
**Corpus:** `uttera-tts-40w` (40 Spanish prompts × ~40 words, UTF-8 text).
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8, Ubuntu 24.04 inside an LXD container sharing the GPU with the host.

## Setup

- Fresh install at `/home/claw/tmp/tts-stt-test/tts-vllm-test` using the repo's `setup.sh`.
- `CACHE_TTL_MINUTES=0` (cache disabled — see below).
- Only this backend was active on the GPU during the bench. vLLM reserved ~22 GB of VRAM at startup (`VLLM_GPU_MEM_UTIL=0.85`, default).
- Bench harness: shared `bench.py` from this repo, `--mode tts`.

## Why cache was disabled

The TTS endpoint caches by the MD5 of `(model, voice, speed, format, params, text)`. The public benchmark corpus only has 40 unique prompts but the burst profiles fire up to 1024 requests — so every prompt gets reused ~25×. With the default cache enabled, the first pass populates and subsequent hits return in ~30 ms (disk read, not model inference). An initial run produced obviously inflated numbers (122 rps @ N=8, 136 rps @ N=1024); those were not throughput measurements, they were cache-read measurements.

Setting `CACHE_TTL_MINUTES=0` bypasses both the write and the read paths, so every request does real TTS inference. That is the apples-to-apples comparison point against Run 5 (which also has cache disabled) and the STT runs (which have no cache).

Production will typically run with the cache on — the cached numbers are real for workloads with repeat-text patterns (IVR menus, notification templates). We simply do not publish them as the model's throughput.

## Commands

```bash
./bench.py --mode tts --server http://127.0.0.1:5100 --profile latency       --corpus ./corpora/uttera-tts-40w --output results/.../vllm-latency.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile burst --n 8   --corpus ./corpora/uttera-tts-40w --output results/.../vllm-burst8.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile burst --n 64  --corpus ./corpora/uttera-tts-40w --output results/.../vllm-burst64.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile burst --n 256 --corpus ./corpora/uttera-tts-40w --output results/.../vllm-burst256.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile burst --n 512 --corpus ./corpora/uttera-tts-40w --output results/.../vllm-burst512.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile burst --n 1024 --corpus ./corpora/uttera-tts-40w --output results/.../vllm-burst1024.json
./bench.py --mode tts --server http://127.0.0.1:5100 --profile sustained --rps 2 --duration 300 --corpus ./corpora/uttera-tts-40w --output results/.../vllm-sustained.json
```

`--rps 2` is the protocol-mandated `0.5 × rps_burst@64` (burst@64 measured 3.08 rps; rounded to 2).

## Raw files in this folder

| File | Content |
|---|---|
| `vllm-latency.{json,csv}` | 20 sequential requests + 3 warmup |
| `vllm-burst{8,64,256,512,1024}.{json,csv}` | N simultaneous requests |
| `vllm-sustained.{json,csv}` | 2 req/s constant for 5 min (600 requests) |

## Observations

- **Zero failures across every profile** — including burst@1024 which completed 1024/1024 OK in 237 s.
- **Throughput plateaus at ~4.2 rps** from N=256 upwards. Below that the per-request pipeline cost dominates.
- **Sustained @ 2 rps / 5 min** stays flat: p95-per-minute `3939, 4032, 4349, 3952, 4211` ms. No drift over the window, no errors, no VRAM growth.
- **Routing header** is always `-` — the vLLM wrapper does not set an `X-Route` header because there is no routing decision to make; the engine handles everything via continuous batching.

## Head-to-head vs Run 5

Same hardware, same corpus, same cache disabled.

| Profile | hotcold RPS | vLLM RPS | vLLM gain | hotcold p50 | vLLM p50 |
|---|---:|---:|---:|---:|---:|
| Latency | 0.23 | 0.48 | **+109 %** | 3 928 ms | 1 795 ms |
| Burst 8 | 0.21 | 0.91 | **+332 %** | 19 346 ms | 3 296 ms |
| Burst 64 | 0.26 | 3.08 | **+1084 %** | 135 s | 11.7 s |
| Burst 256 | 0.26 | 3.98 | **+1431 %** | 308 s | 33.9 s |
| Burst 512 | 0.26 | 4.17 | **+1504 %** | 307 s | 64.2 s |
| Burst 1024 | 0.10 | 4.32 | **+4220 %** | 303 s | 123 s |

vLLM wins everything. The only cost is the fixed 22 GB VRAM reservation — hotcold runs with ~2.5 GB idle and scales up to ~15 GB under load, which is a significant advantage on GPUs shared with other workloads.

## Operational footprint

vLLM keeps the full model + KV cache resident continuously; the GPU is effectively locked to this process for the lifetime of the server. Hotcold's hot worker keeps ~2.5 GB resident; cold workers (another ~2.5 GB each) are spawned on demand and wind down after `COLD_WORKER_IDLE_TIMEOUT`. Which one you want depends on whether the GPU is dedicated to TTS or shared.
