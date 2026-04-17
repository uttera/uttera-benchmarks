# Corpora

Each subdirectory whose name is listed in `PROTOCOL.md §1` is a canonical
corpus. Small ones (prompts, text) are checked in. Big ones (LibriSpeech,
CommonVoice, FLEURS, LJSpeech) are fetched by the scripts below — the
repo stays small, the bytes are reproducible.

## `uttera-tts-40w/` — checked in (40 TXT files, ~40 words each, Spanish)

40 Spanish prompts of ~40 words each, used for TTS burst tests. These
came from `uttera-tts-hotcold/tests/prompts_40w/` and are reproduced here
verbatim so TTS benches do not need a second repo checkout.

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
