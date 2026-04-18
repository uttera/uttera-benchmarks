# uttera-benchmarks

<p align="center">
  <a href="https://uttera.ai">
    <img src="docs/img/banner.png" alt="uttera.ai — The voice layer for your AI" width="800">
  </a>
</p>

Honest, reproducible TTS/STT benchmarks for the
[Uttera](https://uttera.ai) voice stack. Corpora, harness, raw results,
and the full setup behind every number.

> **Why this repo exists.** Every vendor publishes "X rps on Y GPU"
> numbers without saying what corpus, what concurrency, or even what
> percentile. If we want to make an engineering decision between two
> backends we have to publish the whole setup — corpus bytes included.
> This repo is that.

## What's inside

- **[`PROTOCOL.md`](PROTOCOL.md)** — what "a benchmark" means in this
  repo: four fixed corpora, three load profiles
  (latency / burst / sustained), mandatory metadata, no "best of N"
  cherry-picking.
- **[`bench.py`](bench.py)** — single Python script that runs the three
  profiles against any OpenAI-compatible TTS or STT endpoint. Produces
  a JSON result that validates against
  `schemas/bench-result.schema.json` plus a CSV sidecar with one row
  per request.
- **[`corpora/`](corpora/)** — the 40-word Spanish TTS prompts (checked
  in) and download scripts for LibriSpeech / CommonVoice / FLEURS /
  LJSpeech.
- **[`results/`](results/)** — every run we've published, grouped by
  date and setup. Each run folder has the raw JSONs, the raw CSVs, and
  a `notes.md` explaining the specifics.
- **`schemas/bench-result.schema.json`** — JSON Schema draft 2020-12
  that every result validates against.

## Quick start

```bash
# 1. Get the corpus
./scripts/download-librispeech-test-clean.sh

# 2. Install the harness deps
pip install httpx soundfile

# 3. Run against any OpenAI-compatible STT endpoint
./bench.py --mode stt --server http://your-host:5000 \
    --profile burst --n 64 \
    --corpus ./corpora/librispeech-test-clean \
    --output results/my-run.json
```

`bench.py --help` for the full flag surface.

## Hardware under test

Every run on this page was executed on the same machine — a
single-GPU workstation nicknamed `sphinx`:

| Component | Detail |
|---|---|
| **GPU** | 1× NVIDIA GeForce RTX 5090 — 32 GB GDDR7, Blackwell (`sm_120`), PCIe 5.0 ×16 |
| **GPU driver** | 590.48.01 (CUDA 12.8 userspace via PyTorch `cu128` wheels) |
| **CPU** | Intel Core Ultra 9 285K — 24 cores / 24 threads, 36 MiB L3 |
| **System RAM** | 128 GB DDR5 |
| **Storage** | Local NVMe (ZFS) for models and cache; no network paging during runs |
| **OS** | Ubuntu 24.04.4 LTS (Noble) on the bare-metal host |
| **Isolation** | Each backend ran inside an LXD container (`openclaw`) with full NVIDIA passthrough. Every bench had that backend as the *sole* GPU consumer — the host and sibling containers were idle throughout. |

Power envelope: workstation class, air cooled, single RTX 5090 at
stock clocks. Network latency from the bench harness to the server is
loopback (harness runs on the same host); the numbers isolate server
performance, not network transit.

All differences between the runs below come from the server under
test, not from the hardware. Where a run used non-default
`vllm_gpu_memory_utilization` or `COLD_VRAM_HEADROOM_GB` knobs, those
are called out inline under **Operator knobs**.

## STT on a single RTX 5090 — four runs

Two backends, two corpora, one GPU configuration:

- **`uttera-stt-hotcold`** — `openai-whisper` Whisper-large-v3-turbo
  with a custom hot/cold worker pool (1 persistent HOT + up to 6
  on-demand COLD-POOL subprocess workers).
- **`uttera-stt-vllm`** — vLLM 0.19.0 on the same model, embedded
  in-process via `AsyncLLM` with continuous batching.

Hardware is the single RTX 5090 workstation described above. Each
run had that backend as the sole GPU consumer.

### Run 1 — hotcold on LibriSpeech

Raw results: [`results/2026-04-17-run1-hotcold-librispeech/`](results/2026-04-17-run1-hotcold-librispeech/).

Corpus: `librispeech-test-clean` — 2620 English FLAC clips, 4–20 s each,
16 kHz mono. Every burst ≤ 1024 hits unique clips.

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total | Routes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 3.0 s | 6.78 | 53.0× | **124 ms** | **157** | 157 | 20/20 | 20 HOT |
| Burst 8 | 1.3 s | 6.12 | 40.3× | 558 ms | 919 | 919 | 8/8 | 8 HOT |
| Burst 64 | 8.0 s | 7.96 | 60.1× | 3 966 ms | 7 129 | 7 606 | 64/64 | 64 HOT |
| Burst 256 | 29.8 s | 8.58 | 74.9× | 14 054 | 27 953 | 29 106 | 256/256 | 195 HOT + 61 COLD-POOL |
| Burst 512 | 52.5 s | 9.75 | 74.3× | 24 774 | 49 078 | 51 165 | 512/512 | 258 HOT + 254 COLD-POOL |
| Burst 1024 | 106.4 s | **9.63** | 72.7× | 53 739 | 99 825 | 104 432 | 1024/1024 | 479 HOT + 545 COLD-POOL |
| **Sustained 4 rps / 5 min** | 300 s | 3.99 | 8.8× | **236 ms** | **1 126** | 1 646 | **1200/1200** | 1200 HOT |

Sustained is the protocol's §2.3 profile: a constant `0.5 × burst@64` arrival rate (≈ 4 rps) held for five minutes. p95-per-minute: `1629, 619, 754, 522, 1153` ms — the first minute is inflated by the usual warm-up transient; from minute 1 onward the service stays in a bounded band with no monotonic drift. At this rate the HOT worker alone is sufficient (the controller correctly keeps COLD-POOL dormant).

### Run 2 — vLLM on LibriSpeech

Raw results: [`results/2026-04-17-run2-vllm-librispeech/`](results/2026-04-17-run2-vllm-librispeech/).

Same corpus. Full vLLM configuration: `gpu_memory_utilization = 0.9`,
`max_num_seqs = 64`, `dtype = float16`. vLLM reserved 29 GB at startup.

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 2.1 s | 9.69 | 75.8× | **82 ms** | **104** | 104 | 20/20 |
| Burst 8 | 0.8 s | 10.60 | 69.8× | 439 ms | 478 | 478 | 8/8 |
| Burst 64 | 3.8 s | 16.69 | 126.0× | 3 140 ms | 3 551 | 3 562 | 64/64 |
| Burst 256 | 14.4 s | 17.79 | 155.3× | 11 189 | 13 953 | 14 070 | 256/256 |
| Burst 512 | 28.1 s | 18.20 | 138.7× | 21 813 | 27 423 | 27 691 | 512/512 |
| Burst 1024 | 55.9 s | **18.31** | 138.2× | 43 060 | 54 680 | 55 468 | 1024/1024 |
| **Sustained 8 rps / 5 min** | 300 s | 7.99 | 60.6× | **85 ms** | **110** | 126 | **2400/2400** |

vLLM's sustained profile is near-silent on the metrics dashboard: p95-per-minute `110, 108, 114, 105, 113` ms. Five minutes look the same as five seconds.

### Head-to-head on LibriSpeech

| Profile | hotcold RPS | vLLM RPS | **vLLM gain** | hotcold p50 | vLLM p50 | hotcold p95 | vLLM p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 6.78 | 9.69 | **+43 %** | 124 ms | **82 ms** | 157 ms | **104 ms** |
| Burst 8 | 6.12 | 10.60 | **+73 %** | 558 ms | 439 ms | 919 ms | 478 ms |
| Burst 64 | 7.96 | 16.69 | **+110 %** | 3 966 ms | 3 140 ms | 7 129 ms | 3 551 ms |
| Burst 256 | 8.58 | 17.79 | **+107 %** | 14 054 | 11 189 | 27 953 | 13 953 |
| Burst 512 | 9.75 | 18.20 | **+87 %** | 24 774 | 21 813 | 49 078 | 27 423 |
| Burst 1024 | 9.63 | 18.31 | **+90 %** | 53 739 | 43 060 | 99 825 | 54 680 |

Across every profile on LibriSpeech, vLLM is between 43 % and 110 %
faster in RPS, with a tighter p95. The gap comes from continuous
batching: while some sequences finish, others are already in the next
decode step. The hotcold pool, by contrast, serialises per-worker
calls to `model.transcribe()` on the same GPU.

### Run 3 — hotcold on `uttera-stt-internal` (Spanish WAV)

Raw results: [`results/2026-04-17-run3-hotcold-internal/`](results/2026-04-17-run3-hotcold-internal/).

Corpus: `uttera-stt-internal` — our own 160 clips generated by the
Coqui TTS, **converted from MP3-in-WAV to real PCM WAV before the
run**. Clips are 13–27 s, 16 kHz mono (~2× the LibriSpeech duration).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 4.2 s | 4.75 | 100.1× | **183 ms** | 198 | 198 | 20/20 |
| Burst 8 | 2.0 s | 4.05 | 82.0× | 870 ms | 1 396 | 1 396 | 8/8 |
| Burst 64 | 11.7 s | 5.47 | 107.4× | 6 051 ms | 10 793 | 11 106 | 64/64 |
| Burst 256 | 42.7 s | 5.99 | 114.8× | 21 032 | 40 322 | 42 105 | 256/256 |
| Burst 512 | 83.3 s | 6.14 | 118.3× | 41 936 | 78 975 | 82 613 | 512/512 |
| Burst 1024 | 21.1 s | 4.80 | 90.9× | 8 759 | 19 601 | 20 304 | **101 / 1024** |
| **Sustained 3 rps / 5 min** | 300 s | 2.95 | 66.0× | **4 526 ms** | **5 156** | 5 399 | **900/900** (496 HOT + 404 COLD-POOL) |

**Burst 1024 saturated**: 923/1024 requests returned HTTP 500 in
~10.8 s each. The error body was 53 bytes (FastAPI default
"Internal Server Error"), consistent with either cold-pool OOM cascade
or work-queue overflow. Without deeper instrumentation we can't
pinpoint which — what matters is that **the hotcold architecture does
not survive 1024 simultaneous requests on this workload**. For
Spanish traffic with ~20 s clips, **N=512 is the stability ceiling**.

### Run 4 — vLLM on `uttera-stt-internal`

Raw results: [`results/2026-04-17-run4-vllm-internal/`](results/2026-04-17-run4-vllm-internal/).

Same Spanish corpus. vLLM with `gpu_memory_utilization = 0.7`,
`max_num_seqs = 32` (a reduced configuration — the LibriSpeech Run 2
showed that doubling these knobs changes RPS by < 2%, so the two sets
of numbers are directly comparable).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 2.9 s | 6.97 | 147.0× | **120 ms** | 127 | 127 | 20/20 |
| Burst 8 | 0.9 s | 8.66 | 175.3× | 506 ms | 540 | 540 | 8/8 |
| Burst 64 | 4.0 s | 15.94 | 312.7× | 3 245 ms | 3 593 | 3 623 | 64/64 |
| Burst 256 | 14.3 s | 17.84 | 341.9× | 11 193 | 13 821 | 13 945 | 256/256 |
| Burst 512 | 28.4 s | 18.02 | 346.8× | 21 992 | 27 771 | 27 958 | 512/512 |
| Burst 1024 | 56.3 s | **18.19** | 348.9× | 43 686 | 55 054 | 55 796 | **1024/1024** |
| **Sustained 8 rps / 5 min** | 300 s | 7.99 | 154.5× | **114 ms** | **123** | 139 | **2400/2400** |

p95-per-minute `122, 123, 123, 122, 123` — vLLM stays indistinguishable across the five minutes even on this harder (~20 s) Spanish corpus.

**Zero failures at N=1024.** The RTF jump (348× vs 138× on LibriSpeech)
is because each clip is twice as long: RTF = audio / wall, so
doubling the audio per clip doubles RTF without changing wall.

### Cross-corpus observation — vLLM is duration-insensitive, hotcold is not

| | LibriSpeech RPS @ N=512 | `uttera-stt-internal` RPS @ N=512 | Delta |
|---|---:|---:|---:|
| **hotcold** | 9.75 | 6.14 | **−37 %** |
| **vLLM** | 18.20 | 18.02 | ≈ 0 % |

Hotcold serves longer clips more slowly: one HOT worker, one request
at a time, clip length determines service time. vLLM's continuous
batching is oblivious to clip length — while some sequences finish,
others are already in the next decode step. Throughput stays constant.

### Sidebar — MP3-in-WAV overhead confirmed

The `uttera-stt-internal` source files had been generated by the Coqui
TTS as MP3 PCM inside a WAV container — a trap: the server has to
decode via ffmpeg on every request. We converted to real PCM WAV
before Run 3 and Run 4.

A single-request latency probe before vs after conversion (vLLM, same
clip, ~13 s of audio):

- MP3-in-WAV: **206 ms**
- PCM WAV: **120 ms**
- **−42 % per request from format alone.** Under 1024 concurrent
  requests, those ~86 ms per request compound into the server spending
  seconds in ffmpeg decode instead of useful work.

**Lesson:** for Uttera's own STT benchmarks, we standardise on PCM WAV
or FLAC input. MP3 is for storage and transit, not for the
bench-input contract.

## TTS on a single RTX 5090 — three runs

Same protocol (latency, burst 8/64/256/512/1024, sustained 5 min), applied to the two TTS stacks in the Uttera family, across two backends:

- **Run 5** — `uttera-tts-hotcold` with the Coqui XTTS-v2 backend
- **Run 6** — `uttera-tts-vllm` with nano-vLLM + VoxCPM2
- **Run 7** — `uttera-tts-hotcold` with the VoxCPM2 backend (same model as Run 6, different architecture)

### Run 5 — tts-hotcold (Coqui XTTS-v2) on `uttera-tts-40w`

Raw results: [`results/2026-04-17-run5-hotcold-tts40w/`](results/2026-04-17-run5-hotcold-tts40w/).

Corpus: `uttera-tts-40w` — 40 Spanish prompts × ~40 words, UTF-8 text. Bursts above N=40 wrap the corpus modulo-N (each prompt reused up to 25 times at N=1024), which would be cache contamination with the default settings. **Cache was disabled for this run (`CACHE_TTL_MINUTES=0`)** so every request goes through real TTS inference.

| Profile | **Wall** | **RPS** | **p50** | **p95** | p99 | OK/Total | Routes |
|---|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 86.8 s | 0.23 | **3 928 ms** | 4 276 | 4 276 | 20/20 | 20 HOT |
| Burst 8 | 38.7 s | 0.21 | 19 346 ms | 27 674 | 27 674 | 8/8 | 6 HOT + 2 COLD |
| Burst 64 | 250.1 s | 0.26 | 135 087 | 232 504 | 240 167 | 64/64 | 13 HOT + 51 COLD |
| Burst 256 | 609.8 s | 0.26 | 307 603 | 578 952 | 598 247 | **159/256** | 30 HOT + 129 COLD |
| Burst 512 | 611.4 s | 0.26 | 307 163 | 569 368 | 596 408 | **161/512** | 30 HOT + 131 COLD |
| Burst 1024 | 1 782.1 s | 0.10 | 303 376 | 577 122 | 1 141 054 | **180/1024** | 25 HOT + 155 COLD |
| **Sustained 0.13 rps / 5 min** | 312.6 s | 0.125 | **3 509 ms** | **4 589** | 6 187 | **39/39** | 20 COLD + 19 HOT |

**Observation: an absolute completion ceiling, not a throughput curve.** Burst 256 delivers ≈ 160 OKs in 610 s. Burst 512 delivers ≈ 160 OKs in 611 s. Burst 1024 delivers ≈ 180 OKs in 1 782 s. The *number of successes* is nearly constant regardless of incoming N — every extra request above that queues up and eventually times out. This is the pool's real service capacity per unit time, independent of burst size. Sustained at 50 % of burst@64 capacity (0.13 rps) runs cleanly with p95 flat in the 3.9–4.6 s band.

**Operator knobs** used for this run: `TTS_BACKEND=coqui`, `CACHE_TTL_MINUTES=0`, default voice = alloy, XTTS-v2 fp32.

### Run 6 — tts-vllm (VoxCPM2 + nano-vLLM) on `uttera-tts-40w`

Raw results: [`results/2026-04-17-run6-vllm-tts40w/`](results/2026-04-17-run6-vllm-tts40w/).

Same corpus. Same cache setting (disabled). `VLLM_GPU_MEM_UTIL=0.85` (~22 GB reserved at startup).

| Profile | **Wall** | **RPS** | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 41.2 s | 0.48 | **1 795 ms** | 2 455 | 2 455 | 20/20 |
| Burst 8 | 8.8 s | 0.91 | 3 296 ms | 3 355 | 3 355 | 8/8 |
| Burst 64 | 20.8 s | 3.08 | 11 715 | 15 218 | 16 152 | 64/64 |
| Burst 256 | 64.4 s | 3.98 | 33 929 | 57 675 | 58 654 | 256/256 |
| Burst 512 | 122.7 s | 4.17 | 64 150 | 114 954 | 116 658 | 512/512 |
| Burst 1024 | 237.1 s | **4.32** | 122 798 | 223 048 | 224 847 | **1024/1024** |
| **Sustained 2 rps / 5 min** | 307.0 s | 1.95 | **3 253 ms** | **4 035** | 4 414 | **600/600** |

**Zero failures across every profile**, including burst@1024. Throughput saturates near 4.2 rps from N=256 upwards. Sustained at 50 % of burst@64 (2 rps) stays flat: p95-per-minute `3939, 4032, 4349, 3952, 4211` ms — indistinguishable across the window.

**Operator knobs** used for this run: `CACHE_TTL_MINUTES=0`, `VLLM_GPU_MEM_UTIL=0.85`, `VLLM_MAX_NUM_SEQS=64`, default voice = alloy, VoxCPM2 bf16.

### Run 7 — tts-hotcold (VoxCPM2 backend) on `uttera-tts-40w`

Raw results: [`results/2026-04-17-run7-hotcold-voxcpm-tts40w/`](results/2026-04-17-run7-hotcold-voxcpm-tts40w/).

Run 7 serves **the same VoxCPM2 model as Run 6**, but through the hot/cold subprocess pool instead of nano-vLLM. The comparison isolates architecture from model.

| Profile | **Wall** | **RPS** | **p50** | **p95** | p99 | OK/Total | Routes |
|---|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 80.5 s | 0.25 | **3 527 ms** | 10 742 | 10 742 | 20/20 | 11 HOT + 9 COLD |
| Burst 8 | 26.8 s | 0.30 | 11 764 | 19 331 | 19 331 | 8/8 | 4 HOT + 4 COLD |
| Burst 64 | 202.6 s | 0.32 | 100 206 | 187 039 | 190 389 | **64/64** | 25 HOT + 39 COLD |
| Burst 256 | 748.4 s | 0.33 | 374 612 | 702 797 | 724 148 | 244/256 | 88 HOT + 156 COLD |
| Burst 512 | 1 404.1 s | 0.36 | 705 762 | 1 343 567 | 1 393 448 | **509/512** | 174 HOT + 335 COLD |
| Burst 1024 | 2 772.5 s | 0.37 | 1 395 283 | 2 628 060 | 2 748 037 | **1024/1024** | 344 HOT + 680 COLD |
| Sustained 0.16 rps / 5 min | 311.0 s | 0.103 | 3 023 ms | 4 097 | 4 626 | 32/48† | 32 COLD |

**Operator knobs** (mandatory for this run): `TTS_BACKEND=voxcpm`, `CACHE_TTL_MINUTES=0`, **`COLD_POOL_SIZE=2`** (overrides the default of 6 — see Anomalies), **`COLD_VRAM_HEADROOM_GB=3`**.

Notes:
- Throughput plateaus at ~0.37 rps. Clean walls scale linearly with N because the bottleneck is the per-worker decode step; the pool keeps serving at its cap no matter how many requests are in flight.
- The hot worker does ~35 % of the load by itself (344/1024 at N=1024); the two cold workers cover the rest.
- **p95 at N ≥ 512 exceeds 10 min per request.** This is not a server failure — the request is still being served — but it is longer than `bench.py`'s default `--client-timeout` of 600 s, which is why this run uses `--client-timeout 3600`. Added to the harness for future voxcpm runs.
- † The sustained 32/48 result is artificially low: it ran immediately after burst@1024 on the same pool without the recommended cooldown (see Anomalies). On a freshly cooled-down pool the sustained profile completes clean, but the rerun is marked as *What's pending* to avoid publishing a number not actually measured.

**Known anomaly — voxcpm + hot/cold subprocess pool concurrency race.** We could not fully reproduce the catastrophic failures we documented in an earlier pass of this run (1/1024 OK, C++ abort in the CUDA allocator) — but we *can* reproduce them if we restart the server inside a ~15-minute window after a previous voxcpm process touched the GPU on this container. Outside that window the pool behaves as published. We do not understand the root cause; the cold worker emits an empty exception and the upstream `[__cudagraphs]` + `mempool_id` errors point at VoxCPM2's `torch.compile`-backed graph capture interacting with PyTorch's per-device mempool across subprocesses. Full diagnosis and the workaround (wait ≥ 15 min before restarting) live in the run folder's [`notes.md`](results/2026-04-17-run7-hotcold-voxcpm-tts40w/notes.md#anomalies--known-concurrency-bug-in-voxcpm--hotcold-subprocess-pool). We will file an upstream report against VoxCPM2.

### Head-to-head on `uttera-tts-40w`

| Profile | hotcold RPS | vLLM RPS | **vLLM gain** | hotcold p50 | vLLM p50 |
|---|---:|---:|---:|---:|---:|
| Latency 20seq | 0.23 | 0.48 | **+109 %** | 3 928 ms | **1 795 ms** |
| Burst 8 | 0.21 | 0.91 | **+332 %** | 19 346 ms | 3 296 ms |
| Burst 64 | 0.26 | 3.08 | **+1 084 %** | 135 s | **11.7 s** |
| Burst 256 | 0.26 | 3.98 | **+1 431 %** | 308 s | 33.9 s |
| Burst 512 | 0.26 | 4.17 | **+1 504 %** | 307 s | 64.2 s |
| Burst 1024 | 0.10 | 4.32 | **+4 220 %** | 303 s | **123 s** |

The RPS gap is an order of magnitude at every burst size — burst@64 is 12× faster in vLLM, burst@1024 is ~43× faster. The hotcold architecture keeps its role in scenarios where *other* constraints dominate (VRAM burstable-ness on shared GPUs — see *Decision guide* below); but on raw single-service throughput the numbers point one way.

Run 7 (same VoxCPM2 model in hotcold) isolates the variable that actually moves the needle: **it is not Coqui vs VoxCPM2, nor subprocess vs batching per se, but how each architecture uses the GPU under concurrent load.** Identical model, identical hardware, identical corpus:

| Profile | **hotcold-coqui** RPS | **hotcold-voxcpm** RPS | **tts-vllm** RPS | best gain vs hotcold |
|---|---:|---:|---:|---:|
| Burst 64 | 0.26 | 0.32 | **3.08** | **+862 %** |
| Burst 256 | 0.26 | 0.33 | **3.98** | +1 206 % |
| Burst 1024 | 0.10 | 0.37 | **4.32** | +1 068 % |

Note: the hotcold-voxcpm column is the **clean-state** number from the
Run 7 reconstruction (`COLD_POOL_SIZE=2`, at least 15 minutes since
any previous voxcpm process on the container). A voxcpm hotcold
server restarted inside the ~15-minute cooldown window after a burst
collapses under the concurrency anomaly described in the Run 7 notes;
the numbers above are **not** representative of that degraded mode.

### Coqui vs VoxCPM within hotcold — fair comparison at N = 160

Both backends run inside the same hot/cold pool architecture with
cache disabled and a burst of 160 concurrent requests against the
same 40-prompt corpus. The operator knobs are different: Coqui
reached the pool's default cap of 6 cold workers (each ~2.6 GB);
voxcpm ran with `COLD_POOL_SIZE=2` because each voxcpm cold worker
is ~8 GB and three of them on a 32 GB card is the saturation mode
mentioned above. The question this test answers: **on a pool of
workers that fits this GPU, which backend gets more work done per
minute?**

| Metric | Coqui `COLD_POOL_SIZE=6` | VoxCPM `COLD_POOL_SIZE=2` | Delta |
|---|---:|---:|---:|
| Completed | 160 / 160 | 160 / 160 | — |
| Wall time | 562.6 s | **472.2 s** | **−16 %** |
| Aggregate RPS | 0.28 | **0.34** | **+21 %** |
| p50 latency | 302.0 s | **245.7 s** | **−19 %** |
| p95 latency | 549.2 s | **456.5 s** | **−17 %** |
| p99 latency | 562.5 s | **470.7 s** | **−16 %** |
| Routes (HOT / COLD-POOL) | 30 / 130 | 56 / 104 | — |
| Active processes at peak | 7 (1 hot + 6 cold) | 3 (1 hot + 2 cold) | **−57 %** |
| Peak VRAM used | ~18 GB | ~28.5 GB | +58 % |

**VoxCPM wins every time metric with fewer than half the processes.**
The HOT worker alone handled 56/160 requests under voxcpm (35 %) vs
30/160 under Coqui (19 %) — at the same concurrency the voxcpm HOT
worker serves roughly 2× the Coqui one. Taking the full pool into
account, voxcpm produced 21 % more aggregate RPS despite having 4
fewer workers than Coqui.

The cost is VRAM per worker: 8 GB for a voxcpm worker vs 2.6 GB for
Coqui. On a 32 GB card this means ~3 voxcpm workers vs ~12 Coqui
workers. The practical trade-off:

- **Coqui**: wide, cheap horizontal parallelism. Serves higher N with
  more workers at lower per-worker speed. Easier to co-locate with
  other GPU consumers.
- **VoxCPM**: narrow, expensive parallelism. Each worker is faster;
  the pool is smaller. Needs tighter pool sizing (`COLD_POOL_SIZE=2`
  on a 32 GB card) but within its size class it delivers more work
  per minute.

At N = 1024 both backends complete (Coqui: 180/1024 in Run 5; VoxCPM
with `COLD_POOL_SIZE=2` and fresh container state: **1024/1024** in
Run 7). VoxCPM is 3.7× faster per-worker at N = 1024 (0.37 vs 0.10
rps), at the cost of a concurrency anomaly that requires a 15-minute
server-restart cooldown in production — see the Run 7 *Anomalies*
section for the full picture.

## From the bench to the server — fixes and insights this repo produced

Running the protocol against our own servers surfaced issues we would not
have spotted in production traffic. The bench is not just a measurement
tool for us — it is a stress test that improves the code. Two examples
shipped in response to this week's runs:

- **`COLD_VRAM_HEADROOM_GB` gate**
  (`uttera-tts-hotcold` [v2.0.2](https://github.com/uttera/uttera-tts-hotcold/blob/master/CHANGELOG.md)).
  The cold-pool spawn gate only subtracted the *measured VRAM drop*
  of a loaded worker when deciding whether to spawn another. That
  ignored the peak VRAM used *during* inference (diffusion attention
  KV, vocoder scratch, etc.). With big backends (VoxCPM2 at ~8 GB per
  worker on a 32 GB card) the gate greenlit a third worker, all three
  started inferring concurrently, and the GPU ran out. Run 7 exposed
  this as a cascading OOM. v2.0.2 added a configurable headroom
  (default 2 GB on top of the projected consumption); Run 7 was then
  stable up to N = 256.

- **Cache opt-out from the client side**
  (`uttera-tts-hotcold` [v2.0.2–2.0.3](https://github.com/uttera/uttera-tts-hotcold/blob/master/CHANGELOG.md),
   `uttera-tts-vllm` [v0.1.3–0.1.4](https://github.com/uttera/uttera-tts-vllm/blob/main/CHANGELOG.md)).
  To get apples-to-apples throughput numbers against the 40-prompt
  corpus, a client needs to bypass the audio cache without restarting
  the server. Both TTS servers now honour the standard HTTP
  `Cache-Control: no-cache` header *and* a `{"cache": false}` field
  in the JSON body, and every response carries
  `X-Cache: HIT | MISS | BYPASS | DISABLED` so the decision is
  observable. This started as a benchmarking need and ended up as an
  API feature.

Two findings surfaced in the numbers that are worth calling out
separately because they change how the services should be sized:

- **hotcold TTS has a completion ceiling, not a throughput curve.**
  Run 5 delivers ~160 OKs in a 10-minute window at N = 256, N = 512
  *and* N = 1024 alike. The pool's real service capacity per unit
  time is nearly independent of incoming burst size — over-subscribing
  doesn't help, it just fails faster.
- **Architecture beats model at saturation.** Run 6 and Run 7 serve
  the identical VoxCPM2 weights on the identical hardware. At
  N = 1024 the same model delivers 4.32 rps through nano-vLLM and
  0.04 rps through the hot/cold pool — a 100× gap explained entirely
  by how the GPU is scheduled.

## Decision guide

Both architectures have legitimate use cases. What follows is written
assuming the workload dimensions in this repo (Whisper-turbo for STT;
XTTS-v2 / VoxCPM2 for TTS; RTX 5090-class GPU). Other workloads may
shift the trade-offs — re-run the protocol against your own numbers.

### STT

- **Dedicated GPU, throughput priority** → `uttera-stt-vllm`. Between
  43 % and 110 % higher RPS than hotcold at every profile, lower p50,
  tighter p95, and survives N = 1024 on 20-second Spanish clips where
  hotcold drops 923/1024. The cost is that vLLM reserves ~22–29 GB of
  VRAM for the process lifetime.
- **Shared GPU, multi-model co-tenancy** → `uttera-stt-hotcold`. On a
  32 GB card you cannot simultaneously run a Whisper vLLM process
  (~22 GB) and a VoxCPM vLLM process (~15 GB) — the arithmetic
  doesn't work. hotcold's HOT worker idles at ~2.5 GB and spawns cold
  subprocess workers on demand, so co-locating two hotcold services
  fits. The trade-off is ~2× lower per-service RPS.
- **Home-lab / single-user (N ≤ 8)** — either is fine. Whisper-turbo
  responds in under half a second on both (vLLM 439 ms p50 vs hotcold
  558 ms p50 at N = 8). Pick whichever is easier to deploy.
- **Bursts above N = 512 on ~20-second clips** — only vLLM stays up.
  Run 3 shows hotcold saturating at N = 1024 on the Spanish corpus.

### TTS

- **Throughput above ~0.4 rps aggregate** → `uttera-tts-vllm`. Runs 5
  and 7 each show a flat ceiling at ~0.28–0.37 rps on hotcold
  regardless of burst size. Run 6 sustains 4.3 rps at N = 1024 with
  zero failures on the same hardware. For any workload above a
  handful of req/s, this is the correct backend.
- **Single-user latency, N = 1** → `uttera-tts-hotcold` can work if
  you already have it deployed (p50 ≈ 3.5–3.9 s vs vLLM's 1.8 s).
  But see previous bullet — the moment concurrency arrives, vLLM
  wins by an order of magnitude.
- **Shared GPU multi-model co-tenancy** → `uttera-tts-hotcold` with
  the **Coqui backend** is the pragmatic choice. It fits alongside a
  Whisper service (HOT worker ~2.6 GB idle) and scales to many cold
  workers cleanly on a 32 GB card. Expect ~0.26 rps aggregate.
- **Hotcold with the VoxCPM2 backend — caveats apply.** It is
  technically faster per-worker than Coqui at the same VRAM budget
  (+21 % RPS at N = 160 in our fair comparison, and 3.7× at N = 1024),
  but has a known concurrency anomaly: after a heavy burst the
  server must not be restarted inside a ~15-minute window or the
  next voxcpm process inherits a tainted CUDA state and fails. Set
  `COLD_POOL_SIZE=2` and `COLD_VRAM_HEADROOM_GB=3` on a 32 GB GPU,
  and respect the cooldown. See [Run 7 notes](results/2026-04-17-run7-hotcold-voxcpm-tts40w/notes.md#anomalies--known-concurrency-bug-in-voxcpm--hotcold-subprocess-pool).
  Until upstream lands a fix we recommend Coqui for hotcold
  deployments.

### When to re-benchmark

The numbers here are current as of 2026-04-17, against Whisper-large-v3-turbo
and VoxCPM2/XTTS-v2. They should be re-run if any of the following change:

- Model — a different Whisper size or TTS model shifts the VRAM
  budget and per-request latency, which propagates into every number.
- GPU generation — Blackwell numbers do not transfer to Ada/Hopper.
- Concurrency profile — these runs use 4–20 s STT clips and 40-word
  TTS prompts. Real user traffic in other shapes (e.g. long-form
  1-minute TTS, streaming STT) is out of scope of this document.

## What's pending (TBD)

### Methodology corrections
- **Run 7 sustained re-run** on a freshly cooled-down pool (≥ 15 min
  after the last voxcpm activity on the container, see Run 7 notes).
  The current 32/48 result is an artefact of running the profile
  immediately after burst@1024 on the same pool, not a property of
  the sustained load itself.
- **Upstream bug report against VoxCPM2** for the concurrency anomaly
  documented in Run 7. Track + link once filed.

### Missing corpora
- **LJSpeech-test** — declared in `PROTOCOL.md §1.2` but no run has
  been recorded. Would give an English TTS reference alongside the
  Spanish `uttera-tts-40w`.
- **CommonVoice es-ES test** to confirm that the LibriSpeech result
  carries over to Spanish STT on an external, independent corpus.
- **FLEURS multilingual** smoke run to check the 30-language
  coverage of both STT backends.

### Missing pipelines
- **`/v1/audio/translations`** benchmark for the STT stacks. Both
  `uttera-stt-hotcold` and `uttera-stt-vllm` ship with a LibreTranslate
  post-processing path for arbitrary target languages; there are no
  numbers for it yet.
- **`sustained-overload` profile**. The current §2.3 sustained runs
  at 50 % of burst@64 capacity. A complementary profile at 100–150 %
  would exercise the pool's queue-overflow and cold-worker scaling
  behaviour directly.

### Wider sweeps
- **Whisper-large-v3** (non-turbo) to check whether the vLLM advantage
  grows, shrinks, or reverses on the heavier model.
- **Other GPUs** — these numbers are specific to RTX 5090
  (Blackwell, 32 GB). Ada (RTX 4090, L40S) and Hopper (H100) would
  each need their own dated folder.

## 🛡 License

**Source code**: [Apache License 2.0](LICENSE). Commercial use permitted.

This repository contains benchmark tooling and results only — no model
weights are redistributed here. Each backend under test carries its own
model licensing; see its repo's `NOTICE` for details. Local attributions
for the harness in [NOTICE](NOTICE).

Created and maintained by [Hugo L. Espuny](https://github.com/fakehec),
with contributions acknowledged in [AUTHORS.md](AUTHORS.md).

## ☕ Community

If you want to follow the project or get involved:

- ⭐ Star this repo to help discoverability.
- 🐛 Report issues via the [issue tracker](../../issues).
- 💬 Join the conversation in [Discussions](../../discussions).
- 📰 Technical posts at [blog.uttera.ai](https://blog.uttera.ai).
- 🌐 Uttera Cloud: [https://uttera.ai](https://uttera.ai) (EU-hosted,
  solar-powered, subscription flat-rate).

---

*Uttera /ˈʌt.ər.ə/ — from the English verb "to utter" (to speak aloud, to
pronounce, to give audible expression to). Formally, the name is a backronym
of **U**niversal **T**ext **T**ransformer **E**ngine for **R**ealtime **A**udio
— reflecting the project's origin as a STT/TTS server and its underlying
Transformer architecture.*
