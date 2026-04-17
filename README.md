# uttera-benchmarks

Honest, reproducible TTS/STT benchmarks for the [Uttera](https://uttera.ai)
voice stack. Corpora, harness, raw results, and the full story of how
each number was produced — including the mistakes we made and corrected
along the way.

> **Why this repo exists.** Every vendor publishes "X rps on Y GPU"
> numbers without saying what corpus, what concurrency, what GPU
> sharing, or even what percentile. We got burned comparing our own
> numbers across runs that used different clip lengths and accidental
> caching, and decided that if we wanted to make an engineering
> decision we had to publish the whole setup. This repo is that.

## What's inside

- **[`PROTOCOL.md`](PROTOCOL.md)** — what "a benchmark" means in this
  repo: four fixed corpora, three load profiles (latency / burst /
  sustained), mandatory metadata, no "best of N" cherry-picking.
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

## The story so far — STT on a single RTX 5090

Both backends under test share a **single RTX 5090 (32 GB, Blackwell,
CUDA 12.8)** via LXC GPU pass-through:

- **`uttera-stt-hotcold`** at `sphinx:5000` — `openai-whisper`
  Whisper-large-v3-turbo with a custom hot/cold worker pool (1
  persistent HOT + up to 6 on-demand COLD-POOL subprocess workers).
- **`uttera-stt-vllm`** at `openclaw:5002` — vLLM 0.19.0 on the same
  model, embedded in-process via `AsyncLLM` with continuous batching.

Corpus: [**LibriSpeech test-clean**](https://www.openslr.org/12) — 2620
English FLAC clips of 4–20 s each (~5.4 h total). Every burst ≤ 1024
hits unique clips, so no server-side audio-file cache can contaminate
results.

### What we got wrong the first time (and how we found out)

Our first nine-benchmark run used our own 160-clip Spanish TTS-generated
corpus. At N=512 and N=1024 the hotcold numbers swung wildly — sometimes
p50 was 50 s, sometimes 8 s. The raw CSVs showed `route=-` for ~90% of
the fast requests. We initially wrote "cache hit" — **wrong**. The `-`
was actually `COLD-POOL` traffic whose header our bench didn't
recognise. The real contamination had a different root cause:

**Both backends were holding onto VRAM simultaneously.** vLLM pre-reserves
`gpu_memory_utilization × 32 GB = 22 GB` at startup. With hotcold also
alive, free VRAM dropped to ~2 GB, which **prevented hotcold's cold pool
from spawning any subprocess worker**. We published "hotcold doesn't
scale" when the real story was "hotcold was starved of memory by its
neighbour".

We discarded all nine contaminated results. Every number below is from
a run where only one backend was active on the GPU, and the corpus is
LibriSpeech with unique clips for every request.

### Run 1 — hotcold, with vLLM stopped (2026-04-17)

vLLM killed beforehand to free its 22 GB reservation. Only the hotcold
service at `sphinx:5000` was eligible to use the GPU.

Raw results: [`results/2026-04-17-run1-shared-gpu/`](results/2026-04-17-run1-shared-gpu/).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total | Routes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 3.0 s | 6.78 | 53.0× | **124 ms** | **157** | 157 | 20/20 | 20 HOT |
| Burst 8 | 1.3 s | 6.12 | 40.3× | 558 ms | 919 | 919 | 8/8 | 8 HOT |
| Burst 64 | 8.0 s | 7.96 | 60.1× | 3 966 ms | 7 129 | 7 606 | 64/64 | 64 HOT |
| Burst 256 | 29.8 s | 8.58 | 74.9× | 14 054 | 27 953 | 29 106 | 256/256 | 195 HOT + 61 COLD-POOL |
| Burst 512 | 52.5 s | 9.75 | 74.3× | 24 774 | 49 078 | 51 165 | 512/512 | 258 HOT + 254 COLD-POOL |
| Burst 1024 | 106.4 s | **9.63** | 72.7× | 53 739 | 99 825 | 104 432 | 1024/1024 | 479 HOT + 545 COLD-POOL |

### Run 2a — vLLM, with sphinx services alive but idle

vLLM arrived handicapped: `gpu_memory_utilization=0.7, max_num_seqs=32`
(neighbours held ~5 GB, vLLM got 2.28 GB free to reserve on top of the
0.7× fraction). Sphinx services were running but not receiving any
traffic. **Testing the hypothesis that idle neighbours don't cost
compute, only VRAM.**

Raw results: [`results/2026-04-17-run2a-vllm-shared/`](results/2026-04-17-run2a-vllm-shared/).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 2.0 s | 9.80 | 76.7× | **81 ms** | **102** | 102 | 20/20 |
| Burst 8 | 0.8 s | 10.43 | 68.7× | 446 ms | 480 | 480 | 8/8 |
| Burst 64 | 3.8 s | 16.76 | 126.5× | 3 066 ms | 3 491 | 3 534 | 64/64 |
| Burst 256 | 14.2 s | 18.08 | 157.9× | 11 058 | 13 727 | 13 838 | 256/256 |
| Burst 512 | 27.9 s | 18.36 | 139.9× | 21 531 | 27 183 | 27 455 | 512/512 |
| Burst 1024 | 55.7 s | **18.40** | 138.9× | 42 794 | 54 335 | 55 164 | 1024/1024 |

### Run 2b — vLLM, dedicated GPU (sphinx services stopped)

Both `whisper-stt.service` and `coqui-tts.service` on sphinx **stopped**
during the run. vLLM arrived with full production config:
`gpu_memory_utilization=0.9, max_num_seqs=64`. 0.87 GB free at start —
vLLM reserved essentially all 32 GB.

Raw results: [`results/2026-04-17-run2b-vllm-dedicated/`](results/2026-04-17-run2b-vllm-dedicated/).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 2.1 s | 9.69 | 75.8× | **82 ms** | **104** | 104 | 20/20 |
| Burst 8 | 0.8 s | 10.60 | 69.8× | 439 ms | 478 | 478 | 8/8 |
| Burst 64 | 3.8 s | 16.69 | 126.0× | 3 140 ms | 3 551 | 3 562 | 64/64 |
| Burst 256 | 14.4 s | 17.79 | 155.3× | 11 189 | 13 953 | 14 070 | 256/256 |
| Burst 512 | 28.1 s | 18.20 | 138.7× | 21 813 | 27 423 | 27 691 | 512/512 |
| Burst 1024 | 55.9 s | 18.31 | 138.2× | 43 060 | 54 680 | 55 468 | 1024/1024 |

### Hypothesis check: 2a vs 2b

| Metric | 2a (handicapped, neighbours idle) | 2b (full config, GPU dedicated) | Delta |
|---|---:|---:|---:|
| Latency RPS | 9.80 | 9.69 | −1% |
| Burst 8 RPS | 10.43 | 10.60 | +2% |
| Burst 64 RPS | 16.76 | 16.69 | ≈0% |
| Burst 256 RPS | 18.08 | 17.79 | −2% |
| Burst 512 RPS | 18.36 | 18.20 | −1% |
| Burst 1024 RPS | 18.40 | 18.31 | −0.5% |

**All deltas within ±2% — inside measurement noise.** The hypothesis
holds: **idle neighbours don't steal compute, only VRAM.** The
production config (`util=0.9, seqs=64`) does not outperform the
handicapped config (`util=0.7, seqs=32`) because Whisper's `max_model_len`
is only 448 tokens — the KV cache is tiny, 32 concurrent sequences
already saturate the autoregressive decoder, and the extra capacity in
2b has nothing to do.

### Head-to-head: hotcold (Run 1) vs vLLM (Run 2b)

| Profile | hotcold RPS | vLLM RPS | **vLLM gain** | hotcold p50 | vLLM p50 | hotcold p95 | vLLM p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 6.78 | 9.69 | **+43 %** | 124 ms | **82 ms** | 157 ms | **104 ms** |
| Burst 8 | 6.12 | 10.60 | **+73 %** | 558 ms | 439 ms | 919 ms | 478 ms |
| Burst 64 | 7.96 | 16.69 | **+110 %** | 3 966 ms | 3 140 ms | 7 129 ms | 3 551 ms |
| Burst 256 | 8.58 | 17.79 | **+107 %** | 14 054 | 11 189 | 27 953 | 13 953 |
| Burst 512 | 9.75 | 18.20 | **+87 %** | 24 774 | 21 813 | 49 078 | 27 423 |
| Burst 1024 | 9.63 | 18.31 | **+90 %** | 53 739 | 43 060 | 99 825 | 54 680 |

**vLLM wins every metric of every profile.** Single-request latency, tail
latency under burst, aggregate RPS — all favour vLLM on this workload.

### Revised conclusion (replaces the preliminary one from Run 1)

Our preliminary read after Run 1 only said "vLLM hasn't been measured
cleanly yet". Now that it has, the picture for **pure STT throughput**
on a single RTX 5090 is unambiguous:

- **vLLM is ~2× faster** than hotcold in RPS under every burst size and
  **33 % faster** in single-request p50.
- **vLLM has tighter tail latency**: p95 is 1.3–3.6× lower than
  hotcold's at every burst size above 8.
- The vLLM advantage comes from **continuous batching**. The
  hotcold pool spawns subprocess workers that each run an independent
  `model.transcribe()` serialised on the same GPU; that serialisation
  is exactly what continuous batching avoids.

### Revised decision guide — honest version

This is an engineering recommendation, not a sales pitch. The answer
depends on **how you plan to share the GPU**.

#### One STT model per GPU — **use `uttera-stt-vllm`**

If the GPU is dedicated to STT (24 GB+ recommended), vLLM wins on every
metric that matters for a user-facing service. This is the default
recommendation for cloud / multi-tenant deployments.

#### One GPU hosting STT *and* TTS (or multiple models) — **use `uttera-*-hotcold`**

vLLM reserves its `gpu_memory_utilization × VRAM` up front and
**keeps it reserved** for the lifetime of the process. On a single
32 GB GPU, running both an STT and a TTS vLLM process is infeasible:
`22 GB (whisper) + ~15 GB (voxcpm) > 32 GB`.

By contrast, hotcold's HOT worker idles at ~2.5 GB and its COLD-POOL
workers spawn and reap on demand. Running both `uttera-stt-hotcold` +
`uttera-tts-hotcold` on one 32 GB GPU comfortably fits, and throughput
while both are serving traffic simultaneously is limited by the GPU,
not by memory pressure.

The trade-off is real: **roughly 2× lower per-service RPS, in exchange
for hosting two services on the GPU that would otherwise host one**.

#### Home-lab / single-user — either is fine

At N≤8 both backends serve Whisper-large-v3-turbo in under half a second
(vLLM 439 ms p50 vs hotcold 558 ms p50 at N=8). Unless you're
transcribing thousands of clips a day, you will not feel the
difference. Pick whichever is easier to deploy.

### What this changes for Uttera

The original positioning of `uttera-stt-vllm` as "the high-throughput
sibling for large GPUs" is correct but understated. On a dedicated STT
node, vLLM is strictly better for user-facing traffic (lower p50, lower
p95, 2× RPS). The value proposition of `uttera-stt-hotcold` shifts from
"throughput" to **"resource flexibility"**: co-hosting STT and TTS on
one mid-sized GPU in a way vLLM cannot match.

### Run 3 — hotcold, Spanish TTS-generated corpus (`uttera-stt-internal`)

Same `sphinx:5000` backend as Run 1, but the corpus is our own 160
clips from the Coqui TTS, **converted from MP3-in-WAV to real PCM
WAV** before the run. Clips are 13-27 s (2× the LibriSpeech duration).
Raw results: [`results/2026-04-17-run3-hotcold-internal/`](results/2026-04-17-run3-hotcold-internal/).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 4.2 s | 4.75 | 100.1× | **183 ms** | 198 | 198 | 20/20 |
| Burst 8 | 2.0 s | 4.05 | 82.0× | 870 ms | 1 396 | 1 396 | 8/8 |
| Burst 64 | 11.7 s | 5.47 | 107.4× | 6 051 ms | 10 793 | 11 106 | 64/64 |
| Burst 256 | 42.7 s | 5.99 | 114.8× | 21 032 | 40 322 | 42 105 | 256/256 |
| Burst 512 | 83.3 s | 6.14 | 118.3× | 41 936 | 78 975 | 82 613 | 512/512 |
| Burst 1024 | 21.1 s | 4.80 | 90.9× | 8 759 | 19 601 | 20 304 | **101 / 1024** |

**Burst 1024 saturated**: 923/1024 returned HTTP 500 in ~10.8 s each.
The error body was 53 bytes (FastAPI default "Internal Server Error"),
which is consistent with either cold-pool OOM cascade or work-queue
overflow. Without server-side logs we can't pinpoint which — what
matters is that **the hotcold architecture does not survive 1024
simultaneous requests on this workload**. For Spanish production
traffic with ~20 s clips, **N=512 is the stability ceiling**.

### Run 4 — vLLM, same Spanish corpus

Same vLLM config as Run 2a (handicapped `util=0.7, seqs=32`; sphinx
neighbours alive but idle — hypothesis 2a⇔2b already confirmed for
this).
Raw results: [`results/2026-04-17-run4-vllm-internal/`](results/2026-04-17-run4-vllm-internal/).

| Profile | **Wall** | **RPS** | RTF | **p50** | **p95** | p99 | OK/Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency 20seq | 2.9 s | 6.97 | 147.0× | **120 ms** | 127 | 127 | 20/20 |
| Burst 8 | 0.9 s | 8.66 | 175.3× | 506 ms | 540 | 540 | 8/8 |
| Burst 64 | 4.0 s | 15.94 | 312.7× | 3 245 ms | 3 593 | 3 623 | 64/64 |
| Burst 256 | 14.3 s | 17.84 | 341.9× | 11 193 | 13 821 | 13 945 | 256/256 |
| Burst 512 | 28.4 s | 18.02 | 346.8× | 21 992 | 27 771 | 27 958 | 512/512 |
| Burst 1024 | 56.3 s | **18.19** | 348.9× | 43 686 | 55 054 | 55 796 | **1024/1024** |

**Zero failures at N=1024.** The jump in RTF (348× vs 138× on
LibriSpeech) is because each clip is twice as long: RTF = audio /
wall, so doubling audio per clip doubles RTF without changing wall.

### Cross-corpus observation — vLLM is duration-insensitive, hotcold is not

| | Libri RPS @ N=512 | Internal RPS @ N=512 | Delta |
|---|---:|---:|---:|
| **hotcold** | 9.75 | 6.14 | **−37 %** |
| **vLLM** | 18.36 | 18.02 | ≈ 0 % |

Hotcold serves longer clips more slowly: one HOT worker, one request
at a time, clip length determines service time. vLLM's continuous
batching is oblivious to clip length — while some sequences finish,
others are already in the next decode step. Throughput stays constant.

### MP3-in-WAV overhead — confirmed (sidebar)

The original `audio_160/` files had been generated by the Coqui TTS
as MP3 PCM inside a WAV container — a trap: the server has to decode
via ffmpeg on every request. We converted to real PCM WAV before
Run 3 and Run 4.

A single-request latency probe before vs after conversion (vLLM, same
clip, ~13 s of audio):

- MP3-in-WAV: **206 ms**
- PCM WAV: **120 ms**
- **−42 % per request from format alone.** Under 1024 concurrent
  requests, those ~86 ms per request compound into the server spending
  seconds in ffmpeg decode instead of useful work.

**Lesson:** for Uttera's own STT benchmarks, we standardise on PCM WAV
or FLAC input. MP3 is for storage and transit, not for the bench-input
contract.

### Verdict after 4 runs

The decision guide from Run 2b still holds, and Run 3+4 sharpen it:

- **Dedicated STT GPU** → **vLLM**. Now with cross-corpus evidence:
  duration-insensitive, no saturation even at N=1024, 2–4× the RPS of
  hotcold depending on concurrency.
- **Shared GPU (STT + TTS)** → **hotcold**. vLLM's ~22 GB VRAM
  reservation still makes co-location infeasible on a 32 GB GPU.
- **Anything over N=512 at 20-s clips on hotcold is risky.** The node
  saturated at 1024 during Run 3. For any use case that expects
  realistic bursts above that, vLLM is not just faster — it's the only
  one that stays up.

## What's pending (TBD)

- **Sustained-load 5-minute runs** for both backends, to see whether
  p95 drifts over time (memory-leak smell) or stays flat. The 60-second
  sustained runs we did earlier weren't long enough to detect drift.
- **TTS benchmarks**: the same structure against `uttera-tts-hotcold`
  (and `uttera-tts-vllm` if/when it matures past pre-alpha). That
  requires a TTS corpus — the checked-in `corpora/uttera-tts-40w/` is
  ready but no TTS run has been recorded yet.
- **CommonVoice es-ES test** to confirm that the LibriSpeech advantage
  carries over to Spanish (Uttera's main market).
- **Whisper-large-v3** (non-turbo) to check if the vLLM advantage
  grows, shrinks or reverses on the heavier model.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
