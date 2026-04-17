# Run 5 — tts-hotcold on `uttera-tts-40w`

**Date:** 2026-04-17
**Target:** local `uttera-tts-hotcold` v2.0.0, backend **Coqui XTTS-v2** (fp32), on `http://127.0.0.1:5101`.
**Corpus:** `uttera-tts-40w` (40 Spanish prompts × ~40 words, UTF-8 text).
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8, Ubuntu 24.04 inside an LXD container sharing the GPU with the host.

## Setup

- Fresh install at `/home/claw/tmp/tts-stt-test/tts-hotcold-coqui-test` using `./setup.sh coqui`.
- `TTS_BACKEND=coqui`, `CACHE_TTL_MINUTES=0` (cache disabled for apples-to-apples with Run 6 and the STT runs), `NODE_PORT=5101`, `TTS_HOME=/home/claw/.local/share`.
- Only this backend was active on the GPU during the bench (the sphinx host's prod server was stopped to avoid interference).
- Bench harness: shared `bench.py` from this repo, executed with `--mode tts` and the protocol-mandated profiles.

## Commands

```bash
./bench.py --mode tts --server http://127.0.0.1:5101 --profile latency       --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-latency.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 8   --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-burst8.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 64  --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-burst64.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 256 --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-burst256.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 512 --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-burst512.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 1024 --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-burst1024.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile sustained --rps 0.13 --duration 300 --corpus ./corpora/uttera-tts-40w --output results/.../hotcold-sustained.json
```

`--rps 0.13` is the protocol-mandated `0.5 × rps_burst@64` (burst@64 measured 0.26 rps).

## Raw files in this folder

| File | Content |
|---|---|
| `hotcold-latency.{json,csv}` | 20 sequential requests + 3 warmup |
| `hotcold-burst{8,64,256,512,1024}.{json,csv}` | N simultaneous requests |
| `hotcold-sustained.{json,csv}` | 0.13 req/s constant for 5 min (39 requests) |

Each CSV has one row per request: `clip_id, audio_seconds, latency_ms, status, route, bytes, error`. Anyone can recompute p50/p95/p99 from the raw data.

## Observations

- **Single-request latency** is healthy at ~3.9 s for ~40-word Spanish prompts (RTF ≈ 3–4×).
- **Burst @ 8 and 64 complete cleanly** — zero failures, cold-pool workers spin up as expected (at N=64 the split is 13 HOT + 51 COLD).
- **Above N ≈ 160 the node saturates.** The pool manager keeps serving at an aggregate rate of ~0.26 rps (≈160 requests per 10 min), but every excess request times out / 500s:
  - burst@256 → 159/256 OK (38 % fail)
  - burst@512 → 161/512 OK (69 % fail)
  - burst@1024 → 180/1024 OK (82 % fail)
- **Sustained @ 0.13 rps / 5 min** (= 50 % of burst@64 capacity per protocol §2.3): 39/39 OK, p95 stays in a bounded 3.9–4.6 s band, no drift. First minute shows the usual warm-up bump (p95 = 6.2 s → then 4.6 → 4.2 → 3.9 → 4.2).

## Practical ceiling

Coqui + hotcold pool on this single-GPU node **serves ~0.26 rps in aggregate** under concurrent load (confirmed at N=256, 512 and 1024 independently). Anything above that rate will queue up; the first ~160 requests arriving within a 10-minute window complete, the rest either fail fast or time out the client. This is an architectural ceiling of the pool + XTTS-v2 decoder, not a bug.

For an operator: size the target RPS at **≤ 0.15 rps per node** with this configuration, and shard across additional nodes when sustained load exceeds that.

## Head-to-head vs Run 6

Run 6 measured `uttera-tts-vllm` v0.1.2 (nano-vLLM + VoxCPM2) on the same hardware, same corpus, same `CACHE_TTL_MINUTES=0`. vLLM's continuous batching saturates near **4.2 rps** (~16× higher aggregate throughput), with **zero failures at every burst up to N=1024**. The hotcold architecture's advantage is VRAM burstable-ness (the HOT worker alone reserves ~2.5 GB; cold workers spin up only on demand) while vLLM reserves the whole GPU for its KV cache.

## Anomalies

- An earlier sweep on this same server (not published) showed cold_worker processes hanging inside the Coqui `TTS(model_name=...)` ctor during warmup. Root cause was an aborted `setup_assets.sh` that had left a 20 MB truncated `model.pth` under `assets/models/tts/tts_models--multilingual--multi-dataset--xtts_v2/`. `main_tts.py` sets `TTS_HOME=assets/models` on cold-worker spawn; Coqui tried to use the truncated file and blocked silently. Fix: either re-run `setup_assets.sh` to completion, or point `TTS_HOME` to a fully-provisioned cache (we used `/home/claw/.local/share`). The v2.0.0 code itself is sane once the model files are present.
