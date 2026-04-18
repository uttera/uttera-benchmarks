# Corpora

Each subdirectory whose name is listed in `PROTOCOL.md §1` is a canonical
corpus. Small ones (prompts, text) are checked in. Medium ones that are
our own derivative work (TTS-synthesised audio for STT testing) are also
checked in so benchmark WER numbers are reproducible bit-for-bit. Big
public ones (LibriSpeech, CommonVoice, FLEURS, LJSpeech) are fetched by
the scripts below — the repo stays small, the bytes are reproducible.

## `uttera-tts-40w/` — checked in (40 TXT files, ~40 words each, Spanish)

40 Spanish prompts of ~40 words each, used for TTS burst tests. These
came from `uttera-tts-hotcold/tests/prompts_40w/` and are reproduced here
verbatim so TTS benches do not need a second repo checkout.

## `uttera-stt-internal/` — checked in (160 WAV files, ~92 MB, Spanish)

160 Spanish WAV clips — 4 voices × 40 prompts — used as the canonical
STT corpus for Spanish in the Uttera benchmark protocol. Derived by
synthesising the `uttera-tts-40w/` prompts with the Coqui XTTS-v2
backend; the frozen audio bytes are checked in because TTS output is
stochastic and every benchmark run needs the same reference audio to
make WER numbers comparable. See
[`uttera-stt-internal/README.md`](uttera-stt-internal/README.md) for
voice grid, audio format, regeneration notes, and the licensing caveat
inherited from XTTS-v2.

## `librispeech-test-clean/` — download on demand (~350 MB, 2620 FLAC clips)

Canonical English STT reference. Clips are 4–20 s, 16 kHz mono.

```bash
./scripts/download-librispeech-test-clean.sh
```

## `commonvoice-es-test/` — download on demand (~1 GB, Spanish)

Mozilla CommonVoice v17 Spanish test split. Use this for any
Spanish-language STT benchmark.

```bash
./scripts/download-commonvoice-es.sh     # TBD
```

## `fleurs-multilingual/` — download on demand (~3 GB, 102 languages)

Google FLEURS test split. Cross-lingual smoke test.

```bash
./scripts/download-fleurs.sh             # TBD
```

## `ljspeech-test/` — download on demand (~25 MB)

LJSpeech 1.1 metadata test subset (English, single speaker). Used for
TTS sanity.

```bash
./scripts/download-ljspeech.sh           # TBD
```
