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

## STT on a single RTX 5090 — four runs

Two backends, two corpora, one GPU configuration:

- **`uttera-stt-hotcold`** — `openai-whisper` Whisper-large-v3-turbo
  with a custom hot/cold worker pool (1 persistent HOT + up to 6
  on-demand COLD-POOL subprocess workers).
- **`uttera-stt-vllm`** — vLLM 0.19.0 on the same model, embedded
  in-process via `AsyncLLM` with continuous batching.

**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8.

**Every run below was executed with that backend alone on the GPU.**
No other inference workload competed for compute or VRAM during the
measurements.

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

vLLM wins every metric of every profile on LibriSpeech. The advantage
comes from continuous batching: while some sequences finish, others
are already in the next decode step. The hotcold pool, by contrast,
serialises per-worker calls to `model.transcribe()` on the same GPU.

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

## TTS on a single RTX 5090 — two runs

Same protocol (latency, burst 8/64/256/512/1024, sustained 5 min), applied to the two TTS stacks in the Uttera family.

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

**The node saturates at roughly 160 requests per 10-minute window.** Burst 256, 512 and 1024 all show the same ~160 completions regardless of N — everything above that queues up and times out. Sustained at 50 % of burst@64 capacity (0.13 rps) runs cleanly with p95 flat in the 3.9–4.6 s band.

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

### Run 7 — tts-hotcold (VoxCPM2 backend) on `uttera-tts-40w`

Raw results: [`results/2026-04-17-run7-hotcold-voxcpm-tts40w/`](results/2026-04-17-run7-hotcold-voxcpm-tts40w/).

Run 7 serves **the same VoxCPM2 model as Run 6**, but through the hot/cold subprocess pool instead of nano-vLLM. The comparison isolates architecture from model.

| Profile | **Wall** | **RPS** | **p50** | **p95** | p99 | OK/Total | Routes |
|---|---:|---:|---:|---:|---:|---:|---|
| Latency 20seq | 80.5 s | 0.25 | **3 527 ms** | 10 742 | 10 742 | 20/20 | 11 HOT + 9 COLD |
| Burst 8 | 26.8 s | 0.30 | 11 764 | 19 331 | 19 331 | 8/8 | 4 HOT + 4 COLD |
| Burst 64 | 202.9 s | 0.31 | 98 637 | 183 524 | 188 125 | 63/64 | 13 HOT + 51 COLD |
| Burst 256 | 606.4 s | 0.33 | 303 144 | 571 264 | 586 413 | 201/256 | 51 HOT + 149 COLD + 1 COLD→HOT |
| Burst 512 | 109.0 s | 0.28 | 59 756 | 96 941 | 101 209 | **30/512** | 9 HOT + 21 COLD |
| Burst 1024 | 25.5 s | 0.04 | 25 325 | 25 325 | 25 325 | **1/1024** | 1 COLD |
| **Sustained 0.16 rps / 5 min** | 311.0 s | 0.103 | **3 023 ms** | **4 097** | 4 626 | **32/48** | 32 COLD |

Notes:
- `COLD_VRAM_HEADROOM_GB=2` for N ≤ 256, raised to `3` for N = 512/1024 to avoid a cascading-OOM in the pool (see run notes).
- **Collapse at N ≥ 512 is catastrophic**, not graceful: burst@1024 returns 1023 HTTP 500s with empty bodies as cold workers OOM under concurrent inference. Compare to Run 6 (same model, vLLM): 1024/1024 OK.
- The sustained profile shows persistent worker-death residue from the preceding burst@1024 — 16/48 requests dropped even at 0.1 rps until the pool is restarted.

### Head-to-head on `uttera-tts-40w`

| Profile | hotcold RPS | vLLM RPS | **vLLM gain** | hotcold p50 | vLLM p50 |
|---|---:|---:|---:|---:|---:|
| Latency 20seq | 0.23 | 0.48 | **+109 %** | 3 928 ms | **1 795 ms** |
| Burst 8 | 0.21 | 0.91 | **+332 %** | 19 346 ms | 3 296 ms |
| Burst 64 | 0.26 | 3.08 | **+1 084 %** | 135 s | **11.7 s** |
| Burst 256 | 0.26 | 3.98 | **+1 431 %** | 308 s | 33.9 s |
| Burst 512 | 0.26 | 4.17 | **+1 504 %** | 307 s | 64.2 s |
| Burst 1024 | 0.10 | 4.32 | **+4 220 %** | 303 s | **123 s** |

vLLM wins every metric on TTS exactly as it does on STT, and the gap is wider here: burst@64 is 12× faster, burst@1024 is ~43× faster. The hotcold stack still has its place (VRAM burstable-ness on shared GPUs — see *Decision guide* below), but for raw throughput nano-vLLM is in a different class.

Run 7 (same VoxCPM2 model in hotcold) makes the cut even cleaner: it is not *Coqui vs VoxCPM2* or *subprocess vs batching* — identical model, identical hardware, identical corpus. The hotcold pool simply cannot keep a 32 GB GPU busy at voxcpm's size class:

| Profile | **hotcold-coqui** RPS | **hotcold-voxcpm** RPS | **tts-vllm** RPS | best gain vs hotcold |
|---|---:|---:|---:|---:|
| Burst 64 | 0.26 | 0.31 | **3.08** | **+994 %** |
| Burst 256 | 0.26 | 0.33 | **3.98** | +1 206 % |
| Burst 1024 | 0.10 | 0.04 | **4.32** | +10 700 % |

## Decision guide

- **Dedicated STT GPU (24 GB+)** → **vLLM**. 2× the RPS of hotcold at
  most burst sizes, 30 % lower p50, tighter p95, and — critically —
  **survives N=1024 on long Spanish clips where hotcold saturates**.
- **One GPU hosting STT *and* TTS (or multiple models)** → **hotcold**.
  vLLM reserves its `gpu_memory_utilization × VRAM` up front and keeps
  it for the process lifetime. On a 32 GB GPU, running both a whisper
  vLLM process and a VoxCPM vLLM process is infeasible
  (22 GB + ~15 GB > 32 GB). hotcold's HOT worker idles at ~2.5 GB and
  its COLD-POOL spawns on demand, so two hotcold services co-locate
  comfortably. The trade-off is real: roughly 2× lower per-service
  RPS, in exchange for hosting two services on the GPU that would
  otherwise host one.
- **Home-lab / single-user** — either is fine. At N ≤ 8 both backends
  serve Whisper-large-v3-turbo in under half a second (vLLM 439 ms p50
  vs hotcold 558 ms p50 at N=8). Pick whichever is easier to deploy.
- **Anything over N=512 at ~20-s clips on hotcold is risky.** Run 3
  shows the node saturating at N=1024. For use cases expecting
  realistic bursts above that, vLLM is not just faster — it's the
  only one that stays up.
- **TTS at scale** → **vLLM**. The gap widens on TTS: burst@64 is 12×
  faster and burst@1024 is ~43× faster. Hotcold's TTS ceiling lands
  at ~0.26 rps aggregate (≈160 completions per 10 min window) no
  matter how many requests arrive. For any production TTS workload
  above a handful of req/s, use `uttera-tts-vllm`.

## What's pending (TBD)
- **CommonVoice es-ES test** to confirm that the LibriSpeech result
  carries over to Spanish on an external corpus as well.
- **Whisper-large-v3** (non-turbo) to check if the vLLM advantage
  grows, shrinks, or reverses on the heavier model.

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
