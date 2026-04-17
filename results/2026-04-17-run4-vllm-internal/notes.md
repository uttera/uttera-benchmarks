# Run 4 — vLLM on uttera-stt-internal (Spanish WAV)

**Date:** 2026-04-17
**Target:** `stt-node:5000` (`uttera-stt-vllm` v0.1.0, vLLM 0.19.0, Whisper-large-v3-turbo)
**Corpus:** `uttera-stt-internal` (160 WAV PCM 16 kHz mono, 13–27 s each, Spanish).
**Hardware:** 1× NVIDIA RTX 5090 (32 GB, Blackwell), CUDA 12.8

## Setup

Only this backend was active on the GPU during the run. vLLM was started with `gpu_memory_utilization = 0.7`, `max_num_seqs = 32`, `dtype = float16`, `max_model_len = 448`. `vram_free_gb_at_start` at startup: ~3 GB.

## Observations

Numbers are almost indistinguishable from Run 2 (same vLLM, LibriSpeech corpus) from N=64 upwards:

| Profile | Run 2 (LibriSpeech) RPS | Run 4 (Spanish) RPS | Delta |
|---|---:|---:|---:|
| Latency | 9.69 | 6.97 | −28% (clips ~2× longer, expected) |
| Burst 8 | 10.60 | 8.66 | −18% |
| Burst 64 | 16.69 | 15.94 | −5% |
| Burst 256 | 17.79 | 17.84 | ≈ |
| Burst 512 | 18.20 | 18.02 | ≈ |
| Burst 1024 | 18.31 | 18.19 | ≈ |

**Above N=64 vLLM is effectively duration-insensitive.** Longer clips produce more tokens to decode, but continuous batching keeps every decoder step fully utilised; the wall-clock time per request grows proportionally to the tokens generated, which is why p50 scales (3 s at N=64 → 43 s at N=1024) but RPS stays locked around 18.

At low N (latency / burst 8) the per-request time is dominated by the fixed pipeline cost — feature extraction, encoder forward — which the 2× longer clips amplify.

## Anomalies

None. **1024/1024 OK on every profile** — in stark contrast to Run 3 (hotcold), which saturated at N=1024 on this same corpus.

## Operational footprint

vLLM reserved ~22 GB of VRAM continuously for the lifetime of the process. Even idle, those 22 GB are off-limits for any other GPU workload. That is the trade-off of the continuous-batching advantage.
