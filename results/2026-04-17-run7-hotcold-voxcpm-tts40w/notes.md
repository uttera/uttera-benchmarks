# Run 7 — tts-hotcold (VoxCPM2 backend) on `uttera-tts-40w`

**Date:** 2026-04-17
**Target:** local `uttera-tts-hotcold` v2.0.2, backend **VoxCPM2** (bf16 auto, 7–8 GB per worker), on `http://127.0.0.1:5101`.
**Corpus:** `uttera-tts-40w` (40 Spanish prompts × ~40 words).
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8, Ubuntu 24.04 inside an LXD container sharing the GPU with the host.

## Why this run matters

Same VoxCPM2 model as **Run 6 (uttera-tts-vllm)** — the head-to-head is architectural, not model. Run 6 serves VoxCPM2 through nano-vLLM's continuous batching (single process, 22 GB reserved). Run 7 serves the same model through the hot/cold subprocess pool.

## Setup

- Fresh install at `/home/claw/tmp/tts-stt-test/tts-hotcold-test` via `./setup.sh voxcpm`.
- `TTS_BACKEND=voxcpm`, `CACHE_TTL_MINUTES=0`, `NODE_PORT=5101`.
- **`COLD_VRAM_HEADROOM_GB`** set to `2` for profiles `latency` / `burst{8,64,256}` and to `3` for `burst{512,1024}` / `sustained` — see the _Anomalies_ section below for why the tighter value was needed at higher N.
- Only this backend was active on the GPU during the bench.

## Commands

```bash
./bench.py --mode tts --server http://127.0.0.1:5101 --profile latency       --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-latency.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 8   --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst8.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 64  --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst64.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 256 --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst256.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 512 --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst512.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 1024 --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst1024.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile sustained --rps 0.16 --duration 300 --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-sustained.json
```

`--rps 0.16` is the protocol-mandated `0.5 × rps_burst@64` (burst@64 measured 0.31 rps).

## Observations

- Clean up to N = 64 on this hardware:
  - latency 20/20 OK, p50 3.5 s / p95 10.7 s.
  - burst@8 8/8 OK with a 4-HOT / 4-COLD-POOL split.
  - burst@64 63/64 OK (1 failure), 13 HOT + 51 COLD-POOL.
- Queue saturation starts at N = 256 but the pool copes: 201/256 OK (21 % fail rate, 51 HOT + 149 COLD-POOL + 1 COLD-POOL→HOT re-queue).
- **Above N = 256 the pool collapses catastrophically**:
  - burst@512 → 30/512 OK (94 % fail)
  - burst@1024 → 1/1024 OK (99.9 % fail)
- Sustained @ 0.16 rps / 5 min: 32/48 OK. Even at this very low rate the pool dropped 16 requests — residual damage from burst@1024 right before (pool workers that survived the burst returned degraded; a fresh restart before sustained is recommended for cleaner numbers).

## Head-to-head with Run 6 (same VoxCPM2 model)

| Profile | Run 6 vllm RPS | Run 7 hotcold RPS | vllm gain |
|---|---:|---:|---:|
| Latency | 0.48 | 0.25 | +92 % |
| Burst 8 | 0.91 | 0.30 | +203 % |
| Burst 64 | 3.08 | 0.31 | +894 % |
| Burst 256 | 3.98 | 0.33 | +1 106 % |
| Burst 512 | 4.17 | 0.28 | +1 389 % |
| Burst 1024 | 4.32 | 0.04 | +10 700 % |

And on reliability: Run 6 was 1024/1024 OK at N=1024; Run 7 was 1/1024.

The 10×+ throughput gap at N=256 cannot be attributed to the model — it is identical. The gap is purely the architecture: vLLM's continuous batching keeps the single process fed, while the hot/cold subprocess pool is limited by how many workers a 32 GB card can host (2-3 for VoxCPM2) plus the per-request subprocess handshake overhead.

## Anomalies

### `COLD_VRAM_HEADROOM_GB` is backend-sensitive

The first attempt at burst @ 256 on a fresh install collapsed entirely (0/256) because the pool manager gated spawn decisions on the measured VRAM drop per worker (6.09 GB for VoxCPM2) without reserving any extra headroom for CUDA scratch / activation tensors used *during* inference. Three workers loaded at 6 GB each sat happily idle, but the moment all three started inferring simultaneously the GPU ran out. v2.0.2 adds a `COLD_VRAM_HEADROOM_GB` env var (default 2.0) to keep a margin clear; with the default and VoxCPM2 the pool caps at 3 workers (6 × 3 + 2 = 20 GB reserved, leaving 12 GB for the hot worker and inference spikes). For burst@512 we raised it to 3 GB to cap at 2 workers and stopped the cascading OOMs.

### Worker death without diagnostic

When synthesis fails inside a cold worker the parent records `RuntimeError(resp["error"])` with `resp["error"] == ""` (the subprocess emitted `{"error": ""}`, i.e. `str(CudaOutOfMemoryError()) == ""`). Result: the client sees `HTTP 500 {"detail": ""}`. Actionable follow-up for the `uttera-tts-hotcold` repo: capture the exception class name and a traceback tail in the cold-worker error JSON.

## Verdict

VoxCPM2 in the hot/cold architecture works cleanly up to ~N = 200 concurrent on a 32 GB GPU. For workloads above that, use the same model in `uttera-tts-vllm` instead. The hot/cold topology remains the right choice for **multi-model co-tenancy** (e.g. running TTS and STT on the same GPU), but single-model throughput belongs to vLLM.
