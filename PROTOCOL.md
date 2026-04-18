# Uttera benchmark protocol

Canonical reference for how TTS and STT nodes in the Uttera stack are
measured. Any repo that publishes benchmark numbers **must** reference
this document — including a URL, the git commit of this file, and the
command used to produce the numbers — so every claim is reproducible
and every two results can be compared directly.

> **Why this exists.** Before this document, every benchmark we ran used
> slightly different audio durations, different concurrency levels, and
> different metrics. The result was that we could not answer basic
> questions like "did today's change help or hurt?" — each run was an
> island. This protocol makes every benchmark an apples-to-apples
> comparison.

## 1. Fixed corpora

Every benchmark uses **one of these corpora, unmodified, in the order
given**. We never trim, resample, or re-encode them on the fly; that
would invalidate cross-run comparison.

### 1.1 STT corpora

| ID | Source | Content | Use case |
|---|---|---|---|
| **librispeech-test-clean** | [OpenSLR 12](https://www.openslr.org/12) | ~2620 English utterances, 5.4 h total, 4–20 s per clip | Industry-standard reference (NVIDIA Riva, Deepgram, AssemblyAI publish against this) |
| **commonvoice-es-test** | [Mozilla CommonVoice v17 Spanish](https://commonvoice.mozilla.org/en/datasets) test split | ~15 k utterances Spanish, 2–8 s per clip | Spanish-speaking customer realism |
| **fleurs-multilingual** | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) `test` | 102 languages, ~600 clips per language | Multilingual coverage / regression smoke |
| **uttera-stt-internal** | [`corpora/uttera-stt-internal/`](corpora/uttera-stt-internal/) | 160 Spanish WAVs (4 voices × 40 prompts), TTS-synthesised from `uttera-tts-40w` with XTTS-v2 | Spanish STT reference, bit-for-bit reproducible across runs — see corpus README for provenance and XTTS-v2 licence caveat |

### 1.2 TTS corpora

| ID | Source | Content | Use case |
|---|---|---|---|
| **uttera-tts-40w** | `uttera-tts-hotcold/tests/prompts_40w/` | 40 Spanish prompts × ~40 words each | Default in-repo corpus, version-controlled |
| **ljspeech-test** | [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) metadata `test` subset | English, 10–40 words per sentence | Industry reference for single-voice TTS |
| **uttera-tts-internal** | _private_ | Real customer scripts (news, IVR, audiobook chunks) | Internal-only |

All public corpora are hosted on HuggingFace Datasets so every host can
`huggingface-cli download <repo>` and get identical bytes.

## 2. Three mandatory load profiles

Every benchmark reports all three. Single-number results are forbidden.

### 2.1 Latency profile

- **Pattern:** 20 sequential requests, one at a time, after a 3-request warmup.
- **Answers:** "What does a single user experience when no one else is on the node?"
- **Metrics to report:**
  - `lat_single_avg_ms`, `lat_single_p50_ms`, `lat_single_p95_ms`, `lat_single_p99_ms`
  - `rtf_single = audio_duration / wall_time` (TTS: audio_generated / wall; STT: audio_transcribed / wall)

### 2.2 Burst profile

- **Pattern:** N requests fired simultaneously, where **N ∈ {8, 64, 256, 512, 1024}** are all run. Smaller values probe batching ramp-up; larger values probe saturation and tail-latency behaviour.
- **Answers:** "What happens when a batch of requests hits at once?"
- **Metrics to report (per N):**
  - `rps = N / wall_total`
  - `lat_burst_p50`, `lat_burst_p95`, `lat_burst_p99`, `lat_burst_max`
  - `rtf_agg = Σ audio_durations / wall_total`
  - Routing breakdown (for hotcold: HOT / COLD-POOL / COLD-POOL>HOT counts)

### 2.3 Sustained profile

- **Pattern:** A constant arrival rate of **R req/s for 5 minutes**, where R = 0.5 × (the `rps` measured in burst@N=64). This stresses the node at ~50% of its burst capacity.
- **Answers:** "Does the node stay stable under continuous load, or does p95 drift / errors appear?"
- **Metrics to report:**
  - `rps_sustained` (should match R closely; large deviation = saturation)
  - `lat_sustained_p95` at minute 0, 1, 2, 3, 4, 5 — **p95 drift across the window** is the key signal
  - `error_rate_pct`
  - VRAM free at t=0 and t=5 min

## 3. Mandatory metadata

Every benchmark result publishes this block alongside the numbers. A
result without this block is meaningless:

```yaml
node:
  host: node-a
  gpu: NVIDIA RTX 5090 (32 GB, Blackwell)
  cuda: "12.8"
  driver: "580.95.05"
  os: Ubuntu 24.04
service:
  repo: uttera/uttera-stt-vllm
  commit: <git sha>
  protocol_doc: uttera/uttera-infra@<sha>:benchmarks/PROTOCOL.md
  model: openai/whisper-large-v3-turbo
  engine: vllm 0.19.0
  config:
    max_num_seqs: 64
    max_model_len: 448
    gpu_memory_utilization: 0.9
    dtype: float16
    flash_attn: FLASH_ATTN
corpus:
  id: librispeech-test-clean
  n_clips: 400
  total_audio_seconds: 2154
  clip_duration_mean: 5.4
  clip_duration_min: 4.0
  clip_duration_max: 20.0
command: |
  python tests/bench.py --profile burst --n 64 \
    --corpus librispeech-test-clean \
    --server http://127.0.0.1:5000
```

## 4. Reproducibility rules

1. **Warmup before timing.** Every profile begins with 3 throw-away requests.
2. **Same audio, same order, same bytes.** Never randomize the clip order across runs; reproducibility requires determinism.
3. **No aggregate-only reporting.** Publish the raw per-request CSV (one row per request: `clip_id`, `audio_seconds`, `latency_ms`, `status`, `route`) alongside the summary. Anyone must be able to recompute p95 from the raw data.
4. **No "best of N" cherry-picking.** Report the exact run used. If you re-run for variance, report median and range across runs.
5. **Isolate the node.** During the benchmark, no other inference workload shares the GPU. If it must share (dev box), note it explicitly in `node.shared_gpu = true`.
6. **Note the cache state.** First run after boot behaves differently from warm runs because of CUDA graph compilation, kernel selection, and file-system buffer cache. Record `node.cold_start = true/false`.

## 5. Shared bench harness

To enforce the above, each repo's `tests/bench.py` takes the same flags:

```
python tests/bench.py \
    --profile {latency|burst|sustained} \
    --n <N>                       # burst only
    --rps <R>                     # sustained only
    --duration <seconds>          # sustained only, default 300
    --corpus <corpus-id>
    --server <url>
    --output results/<label>.json
```

Output JSON conforms to the metadata schema in §3 plus a `metrics`
sub-document and a `raw_csv_path` pointer. Exact JSON schema lives in
[`schemas/bench-result.schema.json`](schemas/bench-result.schema.json)
(TBD).

## 6. Publication checklist

Before quoting a number in a README, blog post, press release, or
pitch:

- [ ] The JSON result is committed to `benchmarks/results/` in the
      relevant repo and referenced by commit SHA.
- [ ] The metadata block (§3) is complete.
- [ ] The three profiles (§2) were all run — not only the flattering one.
- [ ] The corpus ID (§1) is a public or named-internal corpus, not
      `some_clips_I_had_lying_around/`.
- [ ] If comparing against a competitor's number, the comparison uses the
      **same corpus and same profile**. If that is impossible (their
      benchmark is opaque), say so explicitly.

## 7. Change log of this document

| Date | Author | Change |
|---|---|---|
| 2026-04-16 | J.A.R.V.I.S. | Initial version, drafted against the STT and TTS work done this week. |
