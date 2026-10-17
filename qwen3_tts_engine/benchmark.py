"""Retain per-request SSE audio and compute matched TTS service metrics."""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np


def summarize(records, wall_seconds, frame_rate):
    if wall_seconds <= 0 or frame_rate <= 0:
        raise ValueError("Wall duration and frame rate must be positive")
    valid = [r for r in records if r["success"]]
    return {
        "requests": len(records),
        "successful_requests": len(valid),
        "failed_requests": len(records) - len(valid),
        "wall_seconds": wall_seconds,
        "throughput_frames_s": sum(r["audio_seconds"] for r in valid) * frame_rate / wall_seconds,
        "ttfa_p50_ms": float(np.percentile([r["ttfa_seconds"] * 1000 for r in valid], 50)) if valid else None,
        "ttfa_p95_ms": float(np.percentile([r["ttfa_seconds"] * 1000 for r in valid], 95)) if valid else None,
        "rtf_mean": float(np.mean([r["elapsed_seconds"] / r["audio_seconds"] for r in valid])) if valid else None,
        "request_completion_pass": bool(records) and len(valid) == len(records),
        "quality": "NOT_EVALUATED",
        "server_lifecycle": "NOT_EVALUATED",
        "status": "INCONCLUSIVE" if len(valid) == len(records) else "FAIL",
    }


async def request(session, endpoint, payload, output, index, sample_rate):
    started = time.perf_counter()
    record = {"index": index, "input": payload["input"], "success": False,
              "audio_seconds": 0, "ttfa_seconds": None, "chunks": [], "terminal": None}
    pcm = bytearray()
    terminal = False
    try:
        async with session.post(endpoint, json=payload) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {(await response.text())[:1000]}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("SSE response required to verify successful completion")
            async for raw in response.content:
                if not raw.startswith(b"data:"):
                    continue
                event = json.loads(raw[5:].strip())
                kind = event.get("type")
                if kind == "speech.audio.delta":
                    if terminal:
                        raise ValueError("Audio after terminal event")
                    if event.get("sample_rate", sample_rate) != sample_rate or event.get("channels", 1) != 1:
                        raise ValueError("Unexpected PCM format")
                    chunk = base64.b64decode(event["audio"], validate=True)
                    pcm.extend(chunk)
                    elapsed = time.perf_counter() - started
                    if len(pcm) >= 2 and record["ttfa_seconds"] is None:
                        record["ttfa_seconds"] = elapsed
                    record["chunks"].append({"seconds": elapsed, "bytes": len(chunk)})
                elif kind == "speech.audio.done":
                    if terminal:
                        raise ValueError("Duplicate terminal event")
                    terminal = True
                    record["terminal"] = event
                elif kind in ("speech.audio.error", "error") or "error" in event:
                    raise ValueError(f"Server error: {event}")
                else:
                    raise ValueError(f"Unknown SSE event: {kind}")
        if not terminal or not pcm or len(pcm) % 2:
            raise ValueError("Missing terminal event, empty or malformed PCM")
        samples = np.frombuffer(pcm, dtype="<i2")
        if not np.any(samples):
            raise ValueError("Silent PCM")
        record["success"] = True
        record["audio_seconds"] = len(samples) / sample_rate
        record["peak"] = int(np.abs(samples.astype(np.int32)).max())
        record["clipped_fraction"] = float(np.mean(np.abs(samples.astype(np.int32)) >= 32767))
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, TypeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["elapsed_seconds"] = time.perf_counter() - started
    record["pcm_sha256"] = hashlib.sha256(pcm).hexdigest()
    record["pcm_bytes"] = len(pcm)
    with wave.open(str(output / f"{index:05d}.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    (output / f"{index:05d}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


async def run(options):
    payloads = [json.loads(line) for line in options.workload.read_text().splitlines() if line.strip()]
    if len(payloads) < options.requests:
        raise ValueError("Workload has fewer unique records than requested")
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("input"), str):
            raise ValueError("Each workload record must contain input text")
        payload.update(model=options.model, stream=True, stream_format="sse", response_format="pcm")
    options.output.mkdir(parents=True, exist_ok=False)
    metadata = vars(options).copy()
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}
    metadata["workload_sha256"] = hashlib.sha256(options.workload.read_bytes()).hexdigest()
    (options.output / "runspec.json").write_text(json.dumps(metadata, indent=2) + "\n")
    timeout = aiohttp.ClientTimeout(total=options.timeout)
    connector = aiohttp.TCPConnector(limit=options.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        warmup_dir = options.output / "warmup"
        warmup_dir.mkdir()
        for index in range(options.warmup):
            result = await request(session, options.endpoint, payloads[index % len(payloads)], warmup_dir, index, options.sample_rate)
            if not result["success"]:
                raise RuntimeError(f"Warmup failed: {result}")
        semaphore = asyncio.Semaphore(options.concurrency)

        async def limited(index):
            async with semaphore:
                return await request(session, options.endpoint, payloads[index], options.output, index, options.sample_rate)

        started = time.perf_counter()
        records = await asyncio.gather(*(limited(i) for i in range(options.requests)))
        wall_seconds = time.perf_counter() - started
    metrics = summarize(records, wall_seconds, options.frame_rate)
    (options.output / "results.json").write_text(json.dumps({"metrics": metrics, "requests": records}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["request_completion_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    options = parser.parse_args()
    if min(options.concurrency, options.requests, options.sample_rate) <= 0 or options.warmup < 0:
        parser.error("Counts must be positive, warmup nonnegative")
    if not math.isfinite(options.frame_rate) or options.frame_rate <= 0:
        parser.error("Frame rate must be finite and positive")
    raise SystemExit(asyncio.run(run(options)))


if __name__ == "__main__":
    main()
