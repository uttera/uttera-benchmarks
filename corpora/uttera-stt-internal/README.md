# uttera-stt-internal

160 Spanish WAV clips used as a canonical STT test corpus inside the
Uttera benchmarks protocol. Derived by synthesising the 40 Spanish
prompts in [`../uttera-tts-40w/`](../uttera-tts-40w/) with four
OpenAI-style reference voices via a TTS engine.

**These are TTS-synthesised clips, not real user audio.** The name
`uttera-stt-internal` is retained for continuity with existing
benchmark results (Runs 1–7, published in `../../results/`) and with
`PROTOCOL.md §1.1`, which describes this slot.

## Composition

Grid: 4 voices × 40 prompts = 160 clips

| Voice     | Files                            |
|-----------|----------------------------------|
| `alloy`   | `tts_000_alloy_pNN.wav`   (x40)  |
| `echo`    | `tts_040_echo_pNN.wav`    (x40)  |
| `fable`   | `tts_080_fable_pNN.wav`   (x40)  |
| `nova`    | `tts_120_nova_pNN.wav`    (x40)  |

Filename pattern: `tts_{GLOBAL_IDX:03d}_{VOICE}_p{PROMPT_IDX:02d}.wav`

## Audio format

- Sample rate: 16 kHz (resampled from the 24 kHz XTTS-v2 output for
  STT compatibility).
- Channels: mono
- Bits per sample: 16 (PCM)
- Duration per clip: ~20 s (long enough to exercise chunked
  transcription paths in every Whisper variant).

## Provenance

Generated from `uttera-tts-40w/` prompts using
`uttera-tts-hotcold` with the Coqui XTTS-v2 backend
(`TTS_BACKEND=coqui`, default precision `fp32`, all personality
parameters at their coqui_backend defaults). The generation commit
and timestamps are archived in
`../../results/2026-04-03-run1-spanish-corpus-gen/notes.md`.

Because XTTS-v2 is autoregressive with a temperature, these 160 clips
are **not bit-exact regenerable** — re-running the synthesis produces
audibly-similar but byte-different audio. The frozen corpus in this
directory is therefore the canonical reference: benchmark WER numbers
are computed against transcriptions of **these exact bytes**.

## Why not just regenerate each time

Stochastic TTS output means different generations produce different
word boundaries and pronunciation edge cases. Ground-truth alignment
(our reference transcripts live in `../uttera-tts-40w/` as plain text)
is only meaningful against a frozen audio snapshot. Checking in the
corpus is a small one-time cost (~92 MB) in exchange for
reproducibility of every STT benchmark number published against this
corpus.

## Licence

The audio bytes inherit the licence of the prompts (Apache-2.0, as
for the rest of this repo) plus any licence constraints on the TTS
model used to produce them. Coqui XTTS-v2 weights are under the
[Coqui Public Model License](https://coqui.ai/cpml.txt) — non-commercial.
If you intend commercial use of derivative audio produced by XTTS-v2,
regenerate this corpus with an Apache-2.0-licensed TTS backend (e.g.
VoxCPM2) and replace the bytes in this directory.
