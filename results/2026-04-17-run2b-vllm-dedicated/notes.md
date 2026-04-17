# Run 2b — vLLM, dedicated GPU (sphinx neighbours stopped)

**Date:** 2026-04-17
**Target:** `openclaw:5002` (`uttera-stt-vllm` v0.1.0, vLLM 0.19.0, Whisper-large-v3-turbo)
**Corpus:** `librispeech-test-clean` (2620 FLAC clips, 4-20 s, 16 kHz mono)

## Setup

- Host: `openclaw` LXC on `sphinx`, single RTX 5090 (32 GB, Blackwell, CUDA 12.8).
- `whisper-stt.service` on sphinx: **stopped** for this run.
- `coqui-tts.service` on sphinx: **stopped** for this run.
- vLLM arrived with full production config: `gpu_memory_utilization=0.9, max_num_seqs=64`.
- `vram_free_gb_at_start` reported by vLLM health: **0.87 GB** (vLLM reserved essentially everything for itself — ~29 GB).

## What we set out to test

Companion to Run 2a. User hypothesis: idle neighbours don't eat compute. Comparing Run 2a (handicapped) to Run 2b (full config, GPU dedicated) should show equivalent numbers if the hypothesis is correct.

## Commands

Same as Run 2a but **without** `--shared-gpu` (node.shared_gpu=false in the metadata). Output paths under `results/2026-04-17-run2b-vllm-dedicated/`.

## Result

Run 2a ↔ Run 2b differences are within ±2% across all six profiles. **Hypothesis confirmed**: Whisper inference on vLLM is compute-bound, not KV-cache-bound. Raising `gpu_memory_utilization` from 0.7 to 0.9 and `max_num_seqs` from 32 to 64 does not move the needle because the bottleneck is the autoregressive decoder, not concurrent-sequence capacity. The idle neighbours in Run 2a were not costing performance — they were just holding unused VRAM.

## Anomalies

None. 1884/1884 requests OK.

## Operational note

Both `whisper-stt.service` and `coqui-tts.service` were restarted immediately after the run. Total downtime on sphinx production services: ~6 minutes.
