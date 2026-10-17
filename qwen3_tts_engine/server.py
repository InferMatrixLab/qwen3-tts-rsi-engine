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
        self.failure = None

    async def start(self, app):
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(self.executor, self.model_factory, str(self.options.model), self.options.device)
        self.worker = asyncio.create_task(self.run_continuous() if getattr(self.options, 'continuous', False) else self.run())

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
        return web.json_response({"status": "failed" if self.failure else "ready" if self.model else "starting",
                                  "queued": self.queue.qsize(),
                                  "backend": "standalone",
                                  "audio_delivery": "codec-windows" if getattr(self.options, 'streaming', False)
                                  else "complete-waveform"}, status=503 if self.failure else 200)

    async def run_continuous(self):
        loop = asyncio.get_running_loop()
        pending = {}
        sequence = 0
        tick_index = 0
        interval = getattr(self.options, "prefill_interval", 1)
        while True:
            submissions = []
            cancelled = [identifier for identifier, (_, future, _) in pending.items() if future.cancelled()]
            for identifier in cancelled:
                pending.pop(identifier)
                self.queue.task_done()
            admit_now = not pending or tick_index % interval == 0
            if not pending and not cancelled:
                item = await self.queue.get()
                if item[1].cancelled():
                    self.queue.task_done()
                    continue
                pending[sequence] = item
                submissions.append((sequence, item[0]))
                sequence += 1
            while (admit_now and len(pending) < self.options.max_batch_size and
                   len(submissions) < self.options.prefill_per_tick and not self.queue.empty()):
                item = self.queue.get_nowait()
                if item[1].cancelled():
                    self.queue.task_done()
                    continue
                pending[sequence] = item
                submissions.append((sequence, item[0]))
                sequence += 1

            def emit(identifier, pcm, rate):
                item = pending.get(identifier)
                if item is not None and getattr(self.options, 'streaming', False):
                    item[2].emit(pcm, rate)

            try:
                outputs = await loop.run_in_executor(self.executor,
                    partial(self.model.tick, submissions, emit, cancelled))
                tick_index += 1
                for identifier, output in outputs:
                    _, future, channel = pending.pop(identifier)
                    if not future.done():
                        if channel.error is not None:
                            future.set_exception(channel.error)
                        else:
                            future.set_result(output)
                    self.queue.task_done()
            except asyncio.CancelledError:
                for _, future, channel in pending.values():
                    channel.close()
                    future.cancel()
                raise
            except Exception as error:
                LOGGER.exception('Continuous model failed')
                self.failure = error
                for _, future, channel in pending.values():
                    if not future.done():
                        future.set_exception(error)
                    self.queue.task_done()
                return

    async def speech(self, incoming):
        if self.failure is not None:
            return web.json_response({'error': 'Model worker failed'}, status=503)
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
        try:
            await response.prepare(incoming)
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
    parser.add_argument("--native-gqa", action="store_true")
    parser.add_argument("--text-only-rope", action="store_true")
    parser.add_argument("--prewarm-codec", action="store_true")
    parser.add_argument("--prewarm-prefill", action="store_true")
    parser.add_argument("--triton-kv-update", action="store_true")
    parser.add_argument("--triton-talker-rope", action="store_true")
    parser.add_argument("--codec-chunk-frames", type=int, choices=[8, 16], default=8)
    parser.add_argument("--bucket-kv", action="store_true")
    parser.add_argument("--valid-length-kv", action="store_true")
    parser.add_argument("--graph-embeddings", action="store_true")
    parser.add_argument("--bucket-prefill", action="store_true")
    parser.add_argument("--decode-bucket-step", type=int, default=0)
    parser.add_argument("--batch-audio-transfer", action="store_true")
    parser.add_argument("--reference-single", action="store_true")
    parser.add_argument("--graph-sampling", action="store_true")
    parser.add_argument("--native-rope", action="store_true")
    parser.add_argument("--native-rope-scope", choices=["all", "talker", "predictor"], default="all")
    parser.add_argument("--fused-residual", action="store_true")
    parser.add_argument("--cache-codec-constants", action="store_true")
    parser.add_argument("--codec-code-buffer", action="store_true")
    parser.add_argument("--triton-snake", action="store_true")
    parser.add_argument("--cache-codec-codebooks", action="store_true")
    parser.add_argument("--pack-mlp", action="store_true")
    parser.add_argument("--fused-swiglu", action="store_true")
    parser.add_argument("--rms-norm", choices=["native", "reference-rounding"])
    parser.add_argument("--codec-left-context", type=int, default=72)
    parser.add_argument("--codec-batch-size", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--incremental-codec", action="store_true")
    parser.add_argument("--batch-prefill", action="store_true")
    parser.add_argument("--graph-prefill", action="store_true")
    parser.add_argument("--native-kv-update", action="store_true")
    parser.add_argument("--prefill-per-tick", type=int, default=4)
    parser.add_argument("--prefill-interval", type=int, default=1)
    options = parser.parse_args()
    if not 1 <= options.prefill_interval <= 64 or (options.prefill_interval != 1 and not options.continuous):
        parser.error("--prefill-interval must be 1..64 and requires --continuous when not 1")
    if options.native_rope_scope != "all" and not options.native_rope:
        parser.error("--native-rope-scope requires --native-rope")
    if options.cache_codec_codebooks and not (options.continuous and options.incremental_codec):
        parser.error("--cache-codec-codebooks requires --continuous --incremental-codec")
    if options.triton_snake and not (options.continuous and options.incremental_codec):
        parser.error("--triton-snake requires --continuous --incremental-codec")
    if options.codec_code_buffer and not (options.continuous and options.incremental_codec):
        parser.error("--codec-code-buffer requires --continuous --incremental-codec")
    if options.cache_codec_constants and not (options.continuous and options.incremental_codec):
        parser.error("--cache-codec-constants requires --continuous --incremental-codec")
    if options.fused_residual and options.rms_norm != "native":
        parser.error("--fused-residual requires --rms-norm native")
    if options.incremental_codec and not options.continuous:
        parser.error('--incremental-codec requires --continuous')
    if options.batch_prefill and not options.continuous:
        parser.error('--batch-prefill requires --continuous')
    if options.graph_prefill and not options.continuous:
        parser.error('--graph-prefill requires --continuous')
    if options.prewarm_prefill and (not options.graph_prefill or options.reference_single):
        parser.error('--prewarm-prefill requires --graph-prefill and excludes --reference-single')
    if options.codec_chunk_frames != 8 and (not options.continuous or not options.incremental_codec or options.codec_code_buffer):
        parser.error('--codec-chunk-frames 16 requires continuous incremental codec without --codec-code-buffer')
    if options.triton_talker_rope and not (options.continuous and options.text_only_rope):
        parser.error('--triton-talker-rope requires --continuous --text-only-rope')
    if options.triton_kv_update and not (options.continuous and options.native_kv_update):
        parser.error('--triton-kv-update requires --continuous --native-kv-update')
    if options.graph_sampling and not options.continuous:
        parser.error('--graph-sampling requires --continuous')
    if options.batch_audio_transfer and not options.continuous:
        parser.error('--batch-audio-transfer requires --continuous')
    if options.decode_bucket_step:
        from qwen3_tts_engine.batching import decode_batch_size
        try:
            decode_batch_size(1, options.max_batch_size, options.decode_bucket_step)
        except ValueError as error:
            parser.error(str(error))
        if not options.continuous:
            parser.error('--decode-bucket-step requires --continuous')
    if options.bucket_prefill and not options.graph_prefill:
        parser.error('--bucket-prefill requires --graph-prefill')
    if options.graph_embeddings and not options.continuous:
        parser.error('--graph-embeddings requires --continuous')
    if options.valid_length_kv and (not options.continuous or not options.native_gqa or options.reference_single):
        parser.error("--valid-length-kv requires continuous native GQA without reference-single")
    if options.bucket_kv and not options.continuous:
        parser.error('--bucket-kv requires --continuous')
    if options.native_kv_update and not options.continuous:
        parser.error('--native-kv-update requires --continuous')
    if min(options.max_batch_size, options.queue_size) <= 0 or options.batch_wait_ms < 0 or options.codec_left_context < 0:
        parser.error("Invalid batch or queue settings")
    if options.prefill_per_tick < 1 or (options.continuous and options.max_batch_size & (options.max_batch_size - 1)):
        parser.error('Continuous slots must be a power of two and prefill budget must be positive')
    if options.prewarm_codec and not (options.continuous and options.incremental_codec):
        parser.error('--prewarm-codec requires --continuous --incremental-codec')
    logging.basicConfig(level=logging.INFO)
    factory = partial(CustomVoiceModel, predictor_graph=options.predictor_graph,
                                             talker_graph=options.talker_graph,
                                             rms_norm=options.rms_norm, pack_qkv=options.pack_qkv,
                                             pack_mlp=options.pack_mlp, fused_swiglu=options.fused_swiglu,
                                             native_gqa=options.native_gqa, text_only_rope=options.text_only_rope, reference_single=options.reference_single, native_rope=options.native_rope, native_rope_scope=options.native_rope_scope, fused_residual=options.fused_residual,
                                             codec_left_context=options.codec_left_context,
                                             codec_batch_size=options.codec_batch_size)
    if options.continuous:
        from qwen3_tts_engine.continuous import ContinuousModel
        factory = partial(ContinuousModel, max_slots=options.max_batch_size,
                          rms_norm=options.rms_norm, pack_qkv=options.pack_qkv,
                          pack_mlp=options.pack_mlp, fused_swiglu=options.fused_swiglu,
                          native_gqa=options.native_gqa, text_only_rope=options.text_only_rope, reference_single=options.reference_single, native_rope=options.native_rope, native_rope_scope=options.native_rope_scope, fused_residual=options.fused_residual, graph_prefill=options.graph_prefill,
                          codec_left_context=options.codec_left_context, codec_batch_size=options.codec_batch_size,
                          incremental_codec=options.incremental_codec, batch_prefill=options.batch_prefill,
                          native_kv_update=options.native_kv_update, prewarm_codec=options.prewarm_codec, bucket_kv=options.bucket_kv, graph_embeddings=options.graph_embeddings, bucket_prefill=options.bucket_prefill, decode_bucket_step=options.decode_bucket_step, batch_audio_transfer=options.batch_audio_transfer, graph_sampling=options.graph_sampling, cache_codec_constants=options.cache_codec_constants, codec_code_buffer=options.codec_code_buffer, triton_snake=options.triton_snake, cache_codec_codebooks=options.cache_codec_codebooks, prewarm_prefill=options.prewarm_prefill, triton_kv_update=options.triton_kv_update, triton_talker_rope=options.triton_talker_rope, codec_chunk_frames=options.codec_chunk_frames, valid_length_kv=options.valid_length_kv)
    service = SpeechService(options, factory)
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/health", service.health)
    app.router.add_post("/v1/audio/speech", service.speech)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    web.run_app(app, host=options.host, port=options.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
