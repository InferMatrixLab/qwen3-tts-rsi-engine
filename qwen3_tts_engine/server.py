"""Standalone bounded microbatch speech service with one model owner."""

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging
from pathlib import Path

from aiohttp import web

LOGGER = logging.getLogger(__name__)


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
            active = [(p, f) for p, f in batch if not f.cancelled()]
            groups = {}
            for payload, future in active:
                key = (payload["seed"], payload["max_new_tokens"])
                groups.setdefault(key, []).append((payload, future))
            for group in groups.values():
                try:
                    outputs = await loop.run_in_executor(self.executor, self.model.generate, [p for p, _ in group])
                    if len(outputs) != len(group):
                        raise RuntimeError("Model batch output count mismatch")
                    for (_, future), output in zip(group, outputs):
                        if not future.done():
                            future.set_result(output)
                except Exception as error:
                    LOGGER.exception("Model batch failed")
                    for _, future in group:
                        if not future.done():
                            future.set_exception(error)
            for _ in batch:
                self.queue.task_done()

    async def health(self, incoming):
        return web.json_response({"status": "ready" if self.model else "starting",
                                  "queued": self.queue.qsize(),
                                  "backend": "standalone-reference",
                                  "audio_delivery": "complete-waveform"})

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
        try:
            self.queue.put_nowait((normalized, future))
        except asyncio.QueueFull:
            return web.json_response({"error": "Request queue full"}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(incoming)
        try:
            output = await future
            events = [
                {"type": "speech.audio.delta", "audio": base64.b64encode(output["pcm"]).decode(),
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
    parser.add_argument("--predictor-graph", action="store_true")
    options = parser.parse_args()
    if min(options.max_batch_size, options.queue_size) <= 0 or options.batch_wait_ms < 0:
        parser.error("Invalid batch or queue settings")
    logging.basicConfig(level=logging.INFO)
    service = SpeechService(options, partial(CustomVoiceModel, predictor_graph=options.predictor_graph))
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/health", service.health)
    app.router.add_post("/v1/audio/speech", service.speech)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    web.run_app(app, host=options.host, port=options.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
