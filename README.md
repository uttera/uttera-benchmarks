# uttera-benchmarks

Honest, reproducible TTS/STT benchmarks for the [Uttera](https://uttera.ai)
voice stack. Corpora, harness, raw results, and the full story of how each
number was produced — including the mistakes we made and corrected along
the way.

> **Why this repo exists.** Every vendor publishes "X rps on Y GPU" numbers
> without saying what corpus, what concurrency, what GPU sharing, or even
> what percentile. We got burned comparing our own numbers across runs that
> used different clip lengths and accidental caching, and decided that if
> we wanted to make an engineering decision we had to publish the whole
> setup. This repo is that.

## What's inside

- **[`PROTOCOL.md`](PROTOCOL.md)** — what "a benchmark" means in this repo:
  four fixed corpora, three load profiles (latency / burst / sustained),
  mandatory metadata, no "best of N" cherry-picking.
- **[`bench.py`](bench.py)** — single Python script that runs the three
  profiles against any OpenAI-compatible TTS or STT endpoint. Produces a
  JSON result that validates against `schemas/bench-result.schema.json`
  plus a CSV sidecar with one row per request.
- **[`corpora/`](corpora/)** — the 40-word Spanish TTS prompts (checked in)
  and download scripts for LibriSpeech / CommonVoice / FLEURS / LJSpeech.
- **[`results/`](results/)** — every run we've published, grouped by date
  and setup. Each run folder has the raw JSONs, the raw CSVs, and a
  `notes.md` explaining the specifics.
- **`schemas/bench-result.schema.json`** — JSON Schema draft 2020-12 that
  every result validates against. A result without the full metadata block
  (node, service, corpus, profile, command, metrics) is rejected.

## Quick start

```bash
# 1. Get the corpus
./scripts/download-librispeech-test-clean.sh

# 2. Install the harness's deps (httpx, soundfile)
pip install httpx soundfile

# 3. Run against any OpenAI-compatible STT endpoint
./bench.py --mode stt --server http://your-host:5000 \
    --profile burst --n 64 \
    --corpus ./corpora/librispeech-test-clean \
    --output results/my-run.json
```

See `bench.py --help` for the full flag surface.

## The story so far

### Setup

Both backends under test share a **single RTX 5090 (32 GB, Blackwell, CUDA 12.8)** via LXC GPU pass-through. During the test windows:

- **`uttera-stt-hotcold`** is deployed at `sphinx:5000` running `openai-whisper` Whisper-large-v3-turbo with a custom hot/cold worker pool.
- **`uttera-stt-vllm`** is deployed locally at `openclaw:5002` running `vLLM 0.19.0` on the same model, with continuous batching.

Corpus: [**LibriSpeech test-clean**](https://www.openslr.org/12) — 2620 English FLAC clips of 4–20 s each (~5.4 h of audio). Chosen so that every request in the largest burst (N=1024) hits a unique clip and no server-side audio-file cache can contaminate results.

### What we got wrong the first time (and how we found out)

Our first 9-benchmark run used our own Spanish TTS-generated corpus of 160 clips. At N=512 and N=1024 the hotcold numbers swung wildly between runs — sometimes p50 was 50 s, sometimes 8 s. Investigating the raw CSVs showed the `route` header was `-` (absent) for ~90% of the requests in the fast runs. We initially interpreted this as a server-side file cache — **wrong**. The `-` was actually `COLD-POOL` routing whose header our bench didn't recognise. But meanwhile the real contamination had a different source:

**Both backends were holding onto VRAM simultaneously.** vLLM pre-reserves `gpu_memory_utilization` × total = 22 GB when it starts. With hotcold also alive, free VRAM dropped to ~2 GB, which **prevented hotcold's cold pool from spawning** — it capped at 1 HOT worker regardless of load. The early numbers told us "hotcold doesn't scale" when the real story was "hotcold was starved of memory by its neighbour".

We discarded all nine contaminated results and rebuilt the test. Every number below is from a run where only one backend was active on the GPU.

### Run 1 — hotcold, GPU shared-but-idle with vLLM (2026-04-17)

Setup: vLLM was killed beforehand to free its 22 GB reservation. The
`uttera-stt-hotcold` process at `sphinx:5000` had the GPU to itself for
compute; the TTS service at `sphinx:5100` was alive but idle (no
traffic). Corpus: `librispeech-test-clean` (2620 FLAC, 4–20 s each).

Command, example:

```bash
./bench.py --mode stt --server http://sphinx:5000 \
    --profile burst --n 64 \
    --corpus ./corpora/librispeech-test-clean
```

| Profile | **Wall (s)** | **RPS** | RTF | **p50 (ms)** | **p95 (ms)** | p99 (ms) | OK/Total | Route distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 3.0 | 6.78 | 53.0× | **124** | **157** | 157 | 20/20 | 20 HOT |
| Burst 8 | 1.3 | 6.12 | 40.3× | **558** | 919 | 919 | 8/8 | 8 HOT |
| Burst 64 | 8.0 | 7.96 | 60.1× | 3 966 | 7 129 | 7 606 | 64/64 | 64 HOT |
| Burst 256 | 29.8 | 8.58 | 74.9× | 14 054 | 27 953 | 29 106 | 256/256 | 195 HOT + 61 COLD-POOL |
| Burst 512 | 52.5 | 9.75 | 74.3× | 24 774 | 49 078 | 51 165 | 512/512 | 258 HOT + 254 COLD-POOL |
| Burst 1024 | 106.4 | **9.63** | 72.7× | 53 739 | 99 825 | 104 432 | 1024/1024 | 479 HOT + 545 COLD-POOL |

- **Wall** is the total wall-clock time from "first request sent" to "last request returned".
- **RPS** = successful-requests / wall.
- **RTF (real-time factor)** = total-audio-seconds-processed / wall — how many seconds of audio the node transcribes per second of compute.
- **p50 / p95 / p99** are percentiles of the per-request latency. For a burst, p50 is the median queue-time+inference-time across all N requests; p99 is the near-worst.
- **Route**: `HOT` = served by the single persistent worker; `COLD-POOL` = served by an on-demand subprocess worker from hotcold's dynamic pool, which can spawn up to `COLD_POOL_SIZE` workers (default 6 in this build) while VRAM permits.

### vLLM runs — in progress

Two runs are planned:

1. **Run 2a** — vLLM with `sphinx:5000` + `sphinx:5100` alive but idle.
   This tests whether idle neighbour services actually steal compute
   (working hypothesis: no, they only hold VRAM — no compute contention
   when no traffic is hitting them).
2. **Run 2b** — vLLM with `sphinx:5000` and `sphinx:5100` stopped. vLLM
   arrives at full `gpu_memory_utilization=0.9, max_num_seqs=64` — the
   intended production configuration.

Both will publish the same six profiles against the same corpus.

## Decision guidance (preliminary, will be revised after Run 2)

**Do not cite these numbers as final — the vLLM side is pending.** What the hotcold Run 1 already tells us:

- **Interactive single-request latency is excellent on hotcold**: p50 = **124 ms** for a 14 s clip (RTF ~53× single-request, RTF ~74× aggregate under burst). That is a production-grade p50 for a voice assistant or an IVR.
- **Hotcold scales linearly to the cold-pool cap**: RPS climbs from 6.12 at N=8 to 9.75 at N=512, then plateaus (9.63 at N=1024 = the node is saturated).
- **Tail latency under burst is rough on hotcold**: p95 at N=1024 is almost 100 s. Hotcold uses a FIFO-like queue; if you have bursts of hundreds of concurrent requests, half of them will wait minutes.
- **Cold pool only kicks in from N=256 upwards**. Below that the HOT worker keeps up.

When both runs are published we'll have an honest head-to-head plus a
decision matrix (latency-sensitive vs. throughput-sensitive, single-GPU
vs. multi-GPU, mixed workload vs. dedicated).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
