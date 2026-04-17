# Run 4 — vLLM, Spanish TTS-generated corpus (uttera-stt-internal)

**Date:** 2026-04-17
**Target:** `openclaw:5002` (`uttera-stt-vllm` v0.1.0, vLLM 0.19.0, Whisper-large-v3-turbo)
**Corpus:** `uttera-stt-internal` (160 WAV PCM 16 kHz mono, 13–27 s each, Spanish).

## Setup

- Host: openclaw LXC on sphinx, RTX 5090.
- vLLM arrived with the same handicapped config as Run 2a: `gpu_memory_utilization=0.7, max_num_seqs=32`. Sphinx services (`whisper-stt.service`, `coqui-tts.service`) alive but idle. Run 2a vs 2b confirmed idle neighbours don't cost compute, so we didn't re-stop them.
- `vram_free_gb_at_start`: ~3 GB.

## Observations

Almost indistinguishable from Run 2b (LibriSpeech, dedicated GPU):

| Profile | Run 2b (LibriSpeech) RPS | Run 4 (Spanish) RPS | Delta |
|---|---:|---:|---:|
| Latency | 9.69 | 6.97 | −28 % (clips ~2× longer, expected) |
| Burst 8 | 10.60 | 8.66 | −18 % |
| Burst 64 | 16.69 | 15.94 | −5 % |
| Burst 256 | 17.79 | 17.84 | ≈ |
| Burst 512 | 18.20 | 18.02 | ≈ |
| Burst 1024 | 18.31 | 18.19 | ≈ |

**Above N=64 vLLM is completely duration-insensitive.** Longer clips produce more tokens to decode, but continuous batching keeps every step of the decoder fully utilised; the wall-clock time per request grows proportionally to the tokens generated, which is why p50 scales (3 s at N=64 → 43 s at N=1024) but RPS stays locked around 18.

At low N (latency / burst 8) the per-request time is dominated by a fixed pipeline cost — feature extraction, encoder forward — which the 2× longer clips amplify. That's where the numbers differ from Run 2b.

## Anomalies

None. **1024/1024 OK on every profile** — in stark contrast to Run 3 (hotcold), which saturated at N=1024.

## Operational footprint

vLLM reserved ~22 GB of VRAM continuously (util=0.7 × 32). Even idle, those 22 GB are off-limits for any other GPU workload. That's the other side of the continuous-batching advantage.
