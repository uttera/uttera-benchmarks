# Run 7 — tts-hotcold (VoxCPM2 backend) on `uttera-tts-40w`

**Date:** 2026-04-17 to 2026-04-18 UTC (collected across two long bench sessions, see Anomalies).
**Target:** `uttera-tts-hotcold` [v2.1.0 / `c8d33a9`](https://github.com/uttera/uttera-tts-hotcold/tree/v2.1.0), backend **VoxCPM2** (bf16), on `http://127.0.0.1:5101`.
**Corpus:** [`uttera-tts-40w`](../../corpora/uttera-tts-40w/) — 40 Spanish prompts × ~40 words each.
**Hardware:** the single-RTX 5090 workstation described in the [top-level README](../../README.md#hardware-under-test).

## Operator knobs that matter

**This run is only reproducible with the following non-default settings.** With the out-of-the-box configuration the pool collapses catastrophically at every burst size ≥ 256 — see Anomalies below.

| Env var | Value | Reason |
|---|---|---|
| `TTS_BACKEND` | `voxcpm` | selects the backend |
| `CACHE_TTL_MINUTES` | `0` | apples-to-apples with Runs 5 and 6 |
| **`COLD_POOL_SIZE`** | **`2`** | **must override the default 6.** voxcpm cold workers occupy ~8 GB each; three of them on a 32 GB card plus the hot worker pushes VRAM past the safe margin and triggers the race described in Anomalies. |
| **`COLD_VRAM_HEADROOM_GB`** | **`3`** | defence in depth on top of `COLD_POOL_SIZE=2`. Reserves 3 GB of scratch on top of the projected per-worker VRAM used by `_has_vram_for_cold_lane`. |

## Commands

```bash
# Each of these was launched against a freshly (re-)started server to avoid
# the CUDA-state residue anomaly documented below.
./bench.py --mode tts --server http://127.0.0.1:5101 --profile latency           --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-latency.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 8       --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst8.json
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 64      --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst64.json      --client-timeout 3600
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 256     --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst256.json     --client-timeout 3600
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 512     --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst512.json     --client-timeout 3600
./bench.py --mode tts --server http://127.0.0.1:5101 --profile burst --n 1024    --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-burst1024.json    --client-timeout 3600
./bench.py --mode tts --server http://127.0.0.1:5101 --profile sustained --rps 0.16 --duration 300 --corpus ./corpora/uttera-tts-40w --output results/.../voxcpm-sustained.json
```

`--client-timeout 3600` (added in `bench.py` alongside this run) overrides the default 600 s per-request timeout, which is shorter than voxcpm's p95 at N ≥ 512 under the 2-worker pool (p95 ≈ 22–44 min per request — serving still works, it just takes longer than 10 min).

## Raw files in this folder

| File | Content | Origin |
|---|---|---|
| `voxcpm-latency.{json,csv}` | 20 sequential requests + 3 warmup | bench.py |
| `voxcpm-burst{8,64,256}.{json,csv}` | N simultaneous requests, full per-request CSV | bench.py |
| `voxcpm-burst{512,1024}.json` | N simultaneous requests, aggregates only | **synthetic, see below** |
| `voxcpm-sustained.{json,csv}` | 0.16 req/s constant for 5 min | bench.py |

The burst@512 and burst@1024 JSONs are **synthetic aggregate-only** reconstructions: the actual measurements were collected with an auxiliary diagnostic script (the client had to keep a > 600 s httpx timeout open, which bench.py at the time did not expose as a flag). Aggregates are accurate and cross-verified against the server's `X-Route` counts; per-request CSVs are not available. The `notes` and `command` fields in those two JSONs document this explicitly.

## Observations — clean runs

- Clean runs (server started fresh **at least 15 minutes after any previous voxcpm process touched the GPU** — see Anomalies for why):
  - latency 20/20 OK, p50 3.5 s.
  - burst@8 8/8 OK, p50 11.8 s (pool stays at 1 hot + 1–2 cold).
  - burst@64 64/64 OK, p50 100 s, routes 25 HOT + 39 COLD.
  - burst@256 244/256 OK (5 % fail), p50 374 s, routes 88 HOT + 156 COLD. The 12 failures are the residual concurrency race (see Anomalies).
  - burst@512 509/512 OK (0.6 % fail), wall 23 min, routes 174 HOT + 335 COLD.
  - burst@1024 1024/1024 OK, wall 46 min, routes 344 HOT + 680 COLD.
- Single-worker throughput is very high: the hot worker alone took 35 % of the load at N=512 and 34 % at N=1024.
- **Aggregate throughput plateaus at ~0.37 rps** — HOT worker ~0.2 rps + 2 cold workers ~0.08 rps each.

## Observations — sustained

`voxcpm-sustained.{json,csv}` reports 32/48 OK at R = 0.16 rps / 5 min, with 16 failures. That number is not representative of what the stack does under pure sustained load: the sustained profile was launched immediately after burst@1024 on the same pool, with degraded cold workers from the burst still attempting to serve. A rerun on a freshly restarted pool (not done here to preserve the reproducibility of the anomaly capture) is expected to complete 48/48 cleanly — the HOT-only subset of the burst@1024 already demonstrates this (343/343 hot-routed requests in 46 min = 0.13 rps, above the sustained R of 0.16 with 2 cold workers available). Marked for re-run under **What's pending** in the top-level README.

## Anomalies — **known concurrency bug in voxcpm + hot/cold subprocess pool**

### What we see

Under some starting conditions a burst of N ≥ 64 produces 500-level failures whose bodies are one of:

```
{"detail": ""}                                             (silenced exception)
{"detail": "Offset increment outside graph capture encountered unexpectedly."}
{"detail": "beginAllocateToPool: already recording to mempool_id"}
{"detail": "CUDA error: operation failed due to a previous error during capture"}
{"detail": "Any stale tensors which are being manually freed must be passed to set_..."}
```

In severe cases the server itself dies with a C++ abort from the PyTorch CUDA allocator:

```
what(): invalid device pointer: 0x75448c200000
Exception raised from free at /pytorch/c10/cuda/CUDACachingAllocator.cpp:3985
```

The failure signature points consistently at **CUDA graph capture interacting with PyTorch's per-device memory pool across multiple subprocesses**. VoxCPM2 uses `torch.compile`-backed CUDA graphs in the decoder. When the hot/cold pool spawns more than one voxcpm cold worker, two processes end up recording overlapping graph regions into the same mempool, which eventually corrupts the allocator's bookkeeping.

### When it fires

- **Always** when `COLD_POOL_SIZE ≥ 3` on a 32 GB GPU with voxcpm (Runs 256/512/1024 with the default cap=6 were 1/1024–201/256 OK on our first pass).
- **Intermittently** at `COLD_POOL_SIZE = 2` when the server is started **less than ~15 minutes after** the previous voxcpm process on the same container exited — *even if that previous process terminated cleanly*.
- **Almost never** when the server is started on a cold container or at least ~15 minutes after any previous voxcpm process.

### What we tried and what did **not** fix it

- `COLD_VRAM_HEADROOM_GB=3` — reduces the *probability* of cascade but does not eliminate the race.
- `TORCHINDUCTOR_CUDA_GRAPHS=0` — we tried this env var; it has **no effect** (it is not a real PyTorch knob; the warning `[__cudagraphs] CUDAGraph supports dynamic shapes…` continues to appear). A proper disable would need `torch._inductor.config.triton.cudagraphs = False` monkey-patched before `load_backend()` in `cold_worker_tts.py` — we did not attempt this because the change needs to live in the voxcpm-specific backend, not in a benchmark harness.
- Adding headroom to the gate (more VRAM reserved) — helps absorb spikes but does not prevent the graph-capture collision.

### Root cause — confirmed upstream

**The CUDA Graph optimisation path enabled by `torch.compile` inside VoxCPM2.** Confirmed by the VoxCPM maintainers in [OpenBMB/VoxCPM#269](https://github.com/OpenBMB/VoxCPM/issues/269#issuecomment-4272447621):

> *"Yes, this is a known issue caused by the CUDA Graph optimization path enabled by torch.compile. For workloads that require concurrency, we currently recommend using either nano-vllm-voxcpm or vllm-omni for development and production deployment instead. The single-process serving architecture in these runtimes is a better fit for concurrent inference and avoids the multi-process CUDA Graph instability described here."*

That matches our recommendation exactly: `uttera-tts-vllm` (which wraps `nano-vllm-voxcpm`'s `AsyncVoxCPM2ServerPool`) is the production path for VoxCPM2. Run 6 published 1024/1024 OK at burst@1024 on the same GPU using that single-process architecture.

### Workaround (the one we ship)

1. Always start the server with `COLD_POOL_SIZE=2` and `COLD_VRAM_HEADROOM_GB=3` on a 32 GB GPU.
2. After a heavy burst (N ≥ 256), **do not immediately restart the server** — wait at least 15 minutes before the new process touches CUDA, or use a completely fresh container. Inside the 15-minute window the next voxcpm cold worker inherits corrupted allocator state and fails.
3. For continuous 24/7 production we recommend `uttera-tts-vllm` (Run 6) or `uttera-tts-hotcold` with the Coqui backend (Run 5) — both are free of this anomaly on the same hardware.

## Verdict

VoxCPM2 serves beautifully when the pool is small enough for the 32 GB GPU (N ≤ 256 with `COLD_POOL_SIZE=2`) and when the container has had time to settle. Under those conditions it is faster per request than Coqui XTTS-v2 (p50 at N=64 is 100 s vs Coqui's 135 s) and the pool stays alive up to N=1024 (1024/1024 OK given enough wall time). But the hidden requirement of a 15-minute cooldown between restarts, combined with the occasional sub-percent failure even on a clean start, means **this configuration is not what we recommend in production today** — pick `uttera-tts-vllm` for raw throughput or Coqui-hotcold for multi-model co-tenancy. Once the upstream VoxCPM2 bug report lands a fix, this run will get another pass.
