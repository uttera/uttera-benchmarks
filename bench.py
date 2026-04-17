#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# Canonical Uttera TTS/STT benchmark harness.
#
# One script, three profiles (latency / burst / sustained), both domains.
# TTS and STT are distinguished by --mode (or by an unambiguous --server URL).
# Output: a JSON file that validates against bench-result.schema.json plus a
# per-request CSV sidecar. See PROTOCOL.md for the full spec.
#
# Usage examples:
#
#   # STT burst against a local vllm node
#   ./bench.py --mode stt --server http://localhost:5000 \
#              --profile burst --n 64 \
#              --corpus ./corpora/librispeech-test-clean \
#              --output results/stt-vllm-burst-64.json
#
#   # TTS latency against the hotcold prod
#   ./bench.py --mode tts --server http://sphinx:5100 \
#              --profile latency \
#              --corpus ./corpora/uttera-tts-40w \
#              --output results/tts-hotcold-latency.json
#
#   # STT sustained at 5 rps for 5 minutes
#   ./bench.py --mode stt --server http://localhost:5000 \
#              --profile sustained --rps 5 --duration 300 \
#              --corpus ./corpora/librispeech-test-clean \
#              --output results/stt-vllm-sustained.json
#
# Corpora format: a directory containing either .wav/.mp3/.flac/.ogg clips
# for STT mode, or a directory of .txt files (one prompt per file, UTF-8)
# for TTS mode. The directory name ("librispeech-test-clean", "uttera-tts-40w",
# etc.) must match one of the canonical IDs in PROTOCOL.md §1.
#
# This script is intentionally dependency-light: stdlib + httpx. It is meant
# to run from any host that can reach the server under test.

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

SCHEMA_VERSION = "1.0.0"

STT_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
TTS_EXTS = {".txt"}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    """One unit of work for the benchmark."""
    id: str
    path: Path
    payload: bytes            # raw bytes for STT; UTF-8 text for TTS
    audio_seconds: float      # for STT: duration of the input. For TTS: 0 here
                              # (updated per-request from server's response)


def _load_stt_corpus(corpus_dir: Path) -> list[Clip]:
    import soundfile as sf  # hard dep for STT to read audio duration
    clips: list[Clip] = []
    for p in sorted(corpus_dir.iterdir()):
        if p.suffix.lower() not in STT_EXTS:
            continue
        info = sf.info(str(p))
        clips.append(Clip(
            id=p.name,
            path=p,
            payload=p.read_bytes(),
            audio_seconds=info.frames / info.samplerate,
        ))
    if not clips:
        raise SystemExit(f"No audio files under {corpus_dir} (expected {STT_EXTS}).")
    return clips


def _load_tts_corpus(corpus_dir: Path) -> list[Clip]:
    clips: list[Clip] = []
    for p in sorted(corpus_dir.iterdir()):
        if p.suffix.lower() not in TTS_EXTS:
            continue
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        clips.append(Clip(
            id=p.name,
            path=p,
            payload=text.encode("utf-8"),
            audio_seconds=0.0,
        ))
    if not clips:
        raise SystemExit(f"No prompt files (.txt) under {corpus_dir}.")
    return clips


def _corpus_sha256(clips: list[Clip]) -> str:
    h = hashlib.sha256()
    for c in sorted(clips, key=lambda x: x.id):
        h.update(c.id.encode())
        h.update(b"\x00")
        h.update(hashlib.sha256(c.payload).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

async def _one_stt(client: httpx.AsyncClient, server: str, clip: Clip, model: str) -> dict:
    t0 = time.time()
    try:
        r = await client.post(
            f"{server}/v1/audio/transcriptions",
            files={"file": (clip.id, clip.payload, "application/octet-stream")},
            data={"model": model},
            timeout=600,
        )
        elapsed = time.time() - t0
        return {
            "clip_id": clip.id,
            "audio_seconds": clip.audio_seconds,
            "latency_ms": elapsed * 1000,
            "status": r.status_code,
            "route": r.headers.get("X-Route", "-"),
            "bytes": len(r.content),
        }
    except Exception as e:
        return {
            "clip_id": clip.id, "audio_seconds": clip.audio_seconds,
            "latency_ms": (time.time() - t0) * 1000,
            "status": 0, "route": "-", "bytes": 0, "error": str(e)[:200],
        }


async def _one_tts(client: httpx.AsyncClient, server: str, clip: Clip, model: str) -> dict:
    body = {"model": model, "voice": "alloy",
            "input": clip.payload.decode("utf-8"),
            "response_format": "wav"}
    t0 = time.time()
    try:
        r = await client.post(f"{server}/v1/audio/speech", json=body, timeout=600)
        elapsed = time.time() - t0
        return {
            "clip_id": clip.id,
            # For TTS, audio_seconds is only known from the produced bytes.
            # 22 kHz mono 16-bit PCM would be ~44 kB/s; leave 0 unless the
            # server returned an explicit duration header.
            "audio_seconds": float(r.headers.get("X-Audio-Duration") or 0.0),
            "latency_ms": elapsed * 1000,
            "status": r.status_code,
            "route": r.headers.get("X-Route", "-"),
            "bytes": len(r.content),
        }
    except Exception as e:
        return {
            "clip_id": clip.id, "audio_seconds": 0.0,
            "latency_ms": (time.time() - t0) * 1000,
            "status": 0, "route": "-", "bytes": 0, "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

async def run_latency(mode: str, server: str, clips: list[Clip], model: str,
                      warmup: int, iterations: int) -> list[dict]:
    one = _one_stt if mode == "stt" else _one_tts
    async with httpx.AsyncClient() as client:
        # warmup
        for i in range(warmup):
            await one(client, server, clips[i % len(clips)], model)
        results: list[dict] = []
        for i in range(iterations):
            res = await one(client, server, clips[i % len(clips)], model)
            results.append(res)
    return results


async def run_burst(mode: str, server: str, clips: list[Clip], model: str,
                    warmup: int, n: int) -> list[dict]:
    one = _one_stt if mode == "stt" else _one_tts
    limits = httpx.Limits(max_connections=n + 16, max_keepalive_connections=n + 16)
    async with httpx.AsyncClient(limits=limits) as client:
        for i in range(warmup):
            await one(client, server, clips[i % len(clips)], model)
        tasks = [one(client, server, clips[i % len(clips)], model) for i in range(n)]
        return await asyncio.gather(*tasks)


async def run_sustained(mode: str, server: str, clips: list[Clip], model: str,
                        warmup: int, rps: float, duration_s: int) -> tuple[list[dict], list[float]]:
    """Fire at a constant arrival rate for duration_s. Returns (results, p95_per_minute)."""
    one = _one_stt if mode == "stt" else _one_tts
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
    async with httpx.AsyncClient(limits=limits) as client:
        for i in range(warmup):
            await one(client, server, clips[i % len(clips)], model)

        results: list[dict] = []
        tasks: set[asyncio.Task] = set()
        minute_buckets: list[list[float]] = [[] for _ in range(duration_s // 60 + 1)]

        interval = 1.0 / rps
        t0 = time.time()
        i = 0

        async def dispatch(clip: Clip):
            res = await one(client, server, clip, model)
            results.append(res)
            bucket = min(int((time.time() - t0) // 60), len(minute_buckets) - 1)
            minute_buckets[bucket].append(res["latency_ms"])

        while time.time() - t0 < duration_s:
            clip = clips[i % len(clips)]
            t = asyncio.create_task(dispatch(clip))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            i += 1
            next_fire = t0 + i * interval
            sleep = next_fire - time.time()
            if sleep > 0:
                await asyncio.sleep(sleep)

        await asyncio.gather(*tasks, return_exceptions=True)

        def p95(xs: list[float]) -> float:
            if not xs:
                return 0.0
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(len(xs) * 0.95))]

        return results, [p95(b) for b in minute_buckets]


# ---------------------------------------------------------------------------
# Metrics + metadata
# ---------------------------------------------------------------------------

def _latency_stats(results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == 200]
    if not ok:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "avg": 0}
    lats = sorted(r["latency_ms"] for r in ok)
    def pct(p): return lats[min(len(lats) - 1, int(len(lats) * p))]
    return {
        "min": lats[0], "p50": pct(0.50), "p90": pct(0.90),
        "p95": pct(0.95), "p99": pct(0.99), "max": lats[-1],
        "avg": statistics.mean(lats),
    }


def _routes_histogram(results: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        if r["status"] == 200:
            out[r["route"]] = out.get(r["route"], 0) + 1
    return out


def _detect_repo_commit() -> tuple[str | None, str | None]:
    """Best-effort: detect repo and commit of the *server* we're benching.
    Currently just returns the (repo=None, commit=None). A future version
    can query /health if it exposes git_sha, or accept --service-commit."""
    return None, None


def _gpu_info() -> dict:
    """Try nvidia-smi; fall back to empty."""
    info: dict = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        name, total, free, driver = [x.strip() for x in out.split(",")]
        info["gpu"] = name
        info["vram_total_gb"] = float(total) / 1024
        info["vram_free_gb_at_start"] = float(free) / 1024
        info["driver"] = driver
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query", "--display=COMPUTE"], text=True, timeout=5,
        )
        for line in out.splitlines():
            if "CUDA Version" in line:
                info["cuda"] = line.split(":")[-1].strip()
                break
    except Exception:
        pass
    return info


def _query_server_info(server: str) -> dict:
    try:
        r = httpx.get(f"{server}/health", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Uttera canonical benchmark harness.")
    ap.add_argument("--mode", choices=("stt", "tts"), required=True)
    ap.add_argument("--server", required=True, help="Base URL, e.g. http://localhost:5000")
    ap.add_argument("--profile", choices=("latency", "burst", "sustained"), required=True)
    ap.add_argument("--n", type=int, default=64, help="Burst size (burst only). PROTOCOL recommends N in {8,64,256}.")
    ap.add_argument("--rps", type=float, default=1.0, help="Sustained arrival rate (sustained only).")
    ap.add_argument("--duration", type=int, default=300, help="Sustained duration in seconds (default 300).")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=20, help="Latency profile iterations (default 20).")
    ap.add_argument("--corpus", required=True, type=Path,
                    help="Path to a local corpus directory. Directory name must match a PROTOCOL.md §1 id.")
    ap.add_argument("--model", default="whisper-1",
                    help="Value sent as 'model' form field. Ignored by most servers but required by OpenAI spec.")
    ap.add_argument("--output", type=Path,
                    help="Path to write the JSON result. Default: ./results/<corpus>-<profile>-<stamp>.json")
    ap.add_argument("--tag", default=None, help="Free-form label; copied into the JSON for your own tracking.")
    ap.add_argument("--notes", default=None, help="Free-form note; copied into the JSON.")
    ap.add_argument("--shared-gpu", action="store_true", help="Set node.shared_gpu=true in metadata.")
    ap.add_argument("--cold-start", action="store_true", help="Set node.cold_start=true in metadata.")
    args = ap.parse_args()

    corpus_dir = args.corpus.resolve()
    if not corpus_dir.is_dir():
        ap.error(f"Corpus dir not found: {corpus_dir}")

    corpus_id = corpus_dir.name
    if args.mode == "stt":
        clips = _load_stt_corpus(corpus_dir)
    else:
        clips = _load_tts_corpus(corpus_dir)

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # Run the chosen profile
    sustained_drift: list[float] | None = None
    t_wall_0 = time.time()
    if args.profile == "latency":
        results = asyncio.run(run_latency(args.mode, args.server, clips, args.model,
                                          args.warmup, args.iterations))
        profile_obj = {"kind": "latency", "iterations": args.iterations, "warmup": args.warmup}
    elif args.profile == "burst":
        results = asyncio.run(run_burst(args.mode, args.server, clips, args.model,
                                        args.warmup, args.n))
        profile_obj = {"kind": "burst", "n": args.n, "warmup": args.warmup}
    else:
        results, sustained_drift = asyncio.run(
            run_sustained(args.mode, args.server, clips, args.model,
                          args.warmup, args.rps, args.duration))
        profile_obj = {"kind": "sustained", "rps": args.rps, "duration_s": args.duration, "warmup": args.warmup}
    wall_s = time.time() - t_wall_0
    finished_at = datetime.now(timezone.utc).isoformat()

    ok = [r for r in results if r["status"] == 200]
    fail = [r for r in results if r["status"] != 200]
    # For STT the audio_seconds comes from the input; for TTS it comes from the
    # X-Audio-Duration header (0 if the server doesn't expose it).
    total_audio = sum(r["audio_seconds"] for r in ok)

    # Compose metadata
    node: dict = {"host": platform.node(), "os": f"{platform.system()} {platform.release()}"}
    node.update(_gpu_info())
    node["shared_gpu"] = bool(args.shared_gpu)
    node["cold_start"] = bool(args.cold_start)
    node.setdefault("gpu", "unknown")
    node.setdefault("cuda", "unknown")

    server_health = _query_server_info(args.server)
    service: dict = {
        "repo": "unknown",  # user should fill via --tag or edit afterwards
        "commit": "0000000",
        "protocol_doc": "uttera/uttera-infra:benchmarks/PROTOCOL.md",
        "model": server_health.get("model") or args.model,
        "engine": server_health.get("engine") or "unknown",
        "config": {
            k: server_health.get("metrics", {}).get(k)
            for k in ("max_num_seqs", "max_model_len", "gpu_memory_utilization")
            if server_health.get("metrics", {}).get(k) is not None
        },
    }

    corpus_meta = {
        "id": corpus_id,
        "n_clips": len(clips),
        "total_audio_seconds": sum(c.audio_seconds for c in clips) or total_audio,
        "sha256_manifest": _corpus_sha256(clips),
    }
    if clips and clips[0].audio_seconds:
        durs = [c.audio_seconds for c in clips]
        corpus_meta["clip_duration_mean"] = statistics.mean(durs)
        corpus_meta["clip_duration_min"] = min(durs)
        corpus_meta["clip_duration_max"] = max(durs)

    metrics = {
        "wall_s": wall_s,
        "ok": len(ok),
        "fail": len(fail),
        "rps": len(ok) / wall_s if wall_s > 0 else 0.0,
        "rtf_agg": (total_audio / wall_s) if wall_s > 0 and total_audio > 0 else 0.0,
        "latency_ms": _latency_stats(results),
        "routes": _routes_histogram(results),
    }
    if args.profile == "latency":
        # per-request RTF average, only meaningful single-request
        rtfs = [r["audio_seconds"] / (r["latency_ms"] / 1000.0)
                for r in ok if r["audio_seconds"] and r["latency_ms"] > 0]
        if rtfs:
            metrics["rtf_single_mean"] = statistics.mean(rtfs)
    if sustained_drift is not None:
        metrics["sustained_p95_drift"] = sustained_drift

    # File paths
    if args.output:
        out_json = args.output.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_json = Path.cwd() / "results" / f"{corpus_id}-{args.profile}-{stamp}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_json.with_suffix(".csv")

    # Write raw CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "audio_seconds", "latency_ms", "status", "route", "bytes", "error"])
        for r in results:
            w.writerow([r.get("clip_id", ""), r.get("audio_seconds", 0),
                        r.get("latency_ms", 0), r.get("status", 0),
                        r.get("route", "-"), r.get("bytes", 0),
                        r.get("error", "")])

    # Compose the JSON result
    out: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "node": node,
        "service": service,
        "corpus": corpus_meta,
        "profile": profile_obj,
        "command": " ".join(sys.argv),
        "raw_csv_path": os.path.relpath(out_csv, out_json.parent),
        "metrics": metrics,
    }
    if args.notes:
        out["notes"] = args.notes
    with out_json.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=False)

    # Short human summary on stdout
    print(f"N={len(results)} ok={len(ok)} fail={len(fail)} wall={wall_s:.1f}s "
          f"rps={metrics['rps']:.2f} rtf_agg={metrics['rtf_agg']:.1f}x")
    print(f"lat p50={metrics['latency_ms']['p50']:.0f} "
          f"p95={metrics['latency_ms']['p95']:.0f} "
          f"p99={metrics['latency_ms']['p99']:.0f} ms")
    if sustained_drift:
        print("p95 per minute: " + " ".join(f"{x:.0f}" for x in sustained_drift))
    print(f"\nJSON: {out_json}\nCSV : {out_csv}")


if __name__ == "__main__":
    main()
