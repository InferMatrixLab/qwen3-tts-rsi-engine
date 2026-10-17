"""Standalone bounded microbatch speech service with one model owner."""

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging
from pathlib import Path
import threading

from aiohttp import web

LOGGER = logging.getLogger(__name__)


class AudioChannel:
    """Bound per-client buffering without running HTTP writes on the NPU thread."""

    def __init__(self, loop):
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=16)
        self.closed = threading.Event()
        self.error = None

    def emit(self, pcm, sample_rate):
        if self.closed.is_set():
            return
        pending = asyncio.run_coroutine_threadsafe(self.queue.put((pcm, sample_rate)), self.loop)
        try:
            pending.result(timeout=30)
        except Exception as error:
            pending.cancel()
            self.error = error
            self.closed.set()

    def close(self):
        self.closed.set()
        while not self.queue.empty():
            self.queue.get_nowait()


class SpeechService:
    def __init__(self, options, model_factory):
        self.options = options
        self.model_factory = model_factory
        self.queue = asyncio.Queue(maxsize=options.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-owner")
        self.model = None
        self.worker = None

    async def start(self, app):
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(self.executor, self.model_factory, str(self.options.model), self.options.device)
        self.worker = asyncio.create_task(self.run())

    async def stop(self, app):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        if self.model:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.model.close)
        self.executor.shutdown(wait=True, cancel_futures=True)

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            if self.options.batch_wait_ms:
                await asyncio.sleep(self.options.batch_wait_ms / 1000)
            while len(batch) < self.options.max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            active = [(p, f, c) for p, f, c in batch if not f.cancelled()]
            groups = {}
            for payload, future, channel in active:
                key = (payload["seed"], payload["max_new_tokens"])
                groups.setdefault(key, []).append((payload, future, channel))
            for group in groups.values():
                try:
                    operation = self.model.generate
                    if getattr(self.options, 'streaming', False):
                        def emit(index, pcm, sample_rate):
                            group[index][2].emit(pcm, sample_rate)
                        operation = partial(operation, emit=emit)
                    outputs = await loop.run_in_executor(self.executor, operation, [p for p, _, _ in group])
                    if len(outputs) != len(group):
                        raise RuntimeError("Model batch output count mismatch")
                    for (_, future, channel), output in zip(group, outputs):
                        if not future.done():
                            if channel.error is not None:
                                future.set_exception(channel.error)
                            else:
                                future.set_result(output)
                except Exception as error:
                    LOGGER.exception("Model batch failed")
                    for _, future, _ in group:
                        if not future.done():
                            future.set_exception(error)
            for _ in batch:
                self.queue.task_done()

    async def health(self, incoming):
        return web.json_response({"status": "ready" if self.model else "starting",
                                  "queued": self.queue.qsize(),
                                  "backend": "standalone",
                                  "audio_delivery": "codec-windows" if getattr(self.options, 'streaming', False)
                                  else "complete-waveform"})

    async def speech(self, incoming):
        try:
            payload = await incoming.json()
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            if not isinstance(payload.get("input"), str) or not payload["input"].strip():
                raise ValueError("Nonempty input text is required")
            if len(payload["input"]) > 10000:
                raise ValueError("Input exceeds 10000 characters")
            if payload.get("response_format", "pcm") != "pcm" or payload.get("stream_format", "sse") != "sse":
                raise ValueError("This endpoint requires PCM over SSE")
            normalized = {
                "input": payload["input"], "voice": str(payload.get("voice", "Ryan")).lower(),
                "language": str(payload.get("language", "English")).lower(),
                "instructions": payload.get("instructions", ""),
                "seed": payload.get("seed", 42), "max_new_tokens": payload.get("max_new_tokens", 2048),
            }
            if normalized["voice"] not in self.model.speakers or normalized["language"] not in self.model.languages:
                raise ValueError("Unsupported speaker or language")
            if not isinstance(normalized["instructions"], str):
                raise ValueError("Instructions must be text")
            if type(normalized["seed"]) is not int or not 0 <= normalized["seed"] < 2**32:
                raise ValueError("Seed must be an unsigned 32-bit integer")
            if type(normalized["max_new_tokens"]) is not int or not 2 <= normalized["max_new_tokens"] <= 4096:
                raise ValueError("max_new_tokens must be an integer in [2,4096]")
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        future = asyncio.get_running_loop().create_future()
        channel = AudioChannel(asyncio.get_running_loop())
        try:
            self.queue.put_nowait((normalized, future, channel))
        except asyncio.QueueFull:
            return web.json_response({"error": "Request queue full"}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(incoming)
        try:
            emitted_bytes = 0
            if getattr(self.options, 'streaming', False):
                while True:
                    receive = asyncio.create_task(channel.queue.get())
                    try:
                        done, _ = await asyncio.wait((future, receive), return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        if not receive.done():
                            receive.cancel()
                            await asyncio.gather(receive, return_exceptions=True)
                    if receive in done:
                        pcm, sample_rate = receive.result()
                        event = {"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode(),
                                 "sample_rate": sample_rate, "channels": 1}
                        await response.write(("data: " + json.dumps(event) + "\n\n").encode())
                        emitted_bytes += len(pcm)
                    else:
                        break
            output = await future
            if emitted_bytes > len(output['pcm']):
                raise RuntimeError('Streaming output exceeded final waveform length')
            events = [
                {"type": "speech.audio.delta", "audio": base64.b64encode(output["pcm"][emitted_bytes:]).decode(),
                 "sample_rate": output["sample_rate"], "channels": 1},
                {"type": "speech.audio.done", "finish_reason": output["finish_reason"],
                 "codec_frames": output["codec_frames"]},
            ]
            for event in events:
                await response.write(("data: " + json.dumps(event) + "\n\n").encode())
        except asyncio.CancelledError:
            future.cancel()
            raise
        except ConnectionResetError:
            future.cancel()
        except Exception as error:
            event = {"type": "speech.audio.error", "error": {"message": str(error)}}
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
        finally:
            channel.close()
            if not future.done():
                future.cancel()
        return response


def main():
    from qwen3_tts_engine.model import CustomVoiceModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--queue-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--predictor-graph", nargs="?", const="step", choices=["step", "whole"],
                        help="Experimental predictor graph; requires target-hardware parity validation")
    parser.add_argument("--talker-graph", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--pack-qkv", action="store_true")
    parser.add_argument("--rms-norm", choices=["native", "reference-rounding"])
    options = parser.parse_args()
    if min(options.max_batch_size, options.queue_size) <= 0 or options.batch_wait_ms < 0:
        parser.error("Invalid batch or queue settings")
    logging.basicConfig(level=logging.INFO)
    service = SpeechService(options, partial(CustomVoiceModel, predictor_graph=options.predictor_graph,
                                             talker_graph=options.talker_graph,
                                             rms_norm=options.rms_norm, pack_qkv=options.pack_qkv))
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/health", service.health)
    app.router.add_post("/v1/audio/speech", service.speech)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    web.run_app(app, host=options.host, port=options.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
