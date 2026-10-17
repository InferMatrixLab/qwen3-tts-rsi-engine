import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace

from aiohttp import ClientSession, web

from qwen3_tts_engine.server import SpeechService


def test_service_owns_model_thread_and_validates_requests():
    calls = []

    class Model:
        def __init__(self, path, device):
            self.owner = threading.get_ident()
            self.speakers = ["ryan"]
            self.languages = ["english"]

        def generate(self, payloads):
            assert threading.get_ident() == self.owner
            calls.extend(p["input"] for p in payloads)
            return [{"pcm": b"\x01\x00" * 240, "sample_rate": 24000,
                     "codec_frames": 1, "finish_reason": "stop"} for _ in payloads]

        def close(self):
            assert threading.get_ident() == self.owner
            calls.append("closed")

    async def exercise():
        options = SimpleNamespace(model=Path("unused"), device="unused", queue_size=8,
                                  max_batch_size=2, batch_wait_ms=1)
        service = SpeechService(options, Model)
        app = web.Application()
        app.router.add_post("/speech", service.speech)
        app.on_startup.append(service.start)
        app.on_cleanup.append(service.stop)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}/speech"
        try:
            async with ClientSession() as session:
                async with session.post(url, json=[]) as response:
                    assert response.status == 400
                async with session.post(url, json={"input": "test", "max_new_tokens": -1}) as response:
                    assert response.status == 400
                async with session.post(url, json={"input": "test"}) as response:
                    assert response.status == 200
                    text = await response.text()
                    assert "speech.audio.delta" in text
                    assert "speech.audio.done" in text
        finally:
            await runner.cleanup()
        assert calls == ["test", "closed"]

    asyncio.run(exercise())


def test_streaming_first_audio_precedes_model_completion_and_tail_is_not_duplicated():
    import base64
    import json

    received_first = threading.Event()
    prefix = b'\x01\x00' * 120
    tail = b'\x02\x00' * 120

    class Model:
        def __init__(self, path, device):
            self.speakers = ['ryan']
            self.languages = ['english']

        def generate(self, payloads, emit):
            emit(0, prefix, 24000)
            if not received_first.wait(timeout=3):
                raise RuntimeError('Audio was buffered until model completion')
            return [{'pcm': prefix + tail, 'sample_rate': 24000,
                     'codec_frames': 2, 'finish_reason': 'stop'}]

        def close(self):
            pass

    async def exercise():
        options = SimpleNamespace(model=Path('unused'), device='unused', queue_size=8,
                                  max_batch_size=1, batch_wait_ms=0, streaming=True)
        service = SpeechService(options, Model)
        app = web.Application()
        app.router.add_post('/speech', service.speech)
        app.on_startup.append(service.start)
        app.on_cleanup.append(service.stop)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        try:
            async with ClientSession() as session:
                async with session.post(f'http://127.0.0.1:{runner.addresses[0][1]}/speech',
                                        json={'input': 'test'}) as response:
                    first = json.loads((await response.content.readline()).decode()[6:])
                    assert base64.b64decode(first['audio']) == prefix
                    received_first.set()
                    rest = await response.text()
                    events = [json.loads(line[6:]) for line in rest.splitlines() if line.startswith('data: ')]
                    assert base64.b64decode(events[0]['audio']) == tail
                    assert events[1]['type'] == 'speech.audio.done'
        finally:
            received_first.set()
            await runner.cleanup()

    asyncio.run(exercise())


def test_continuous_service_admits_replacement_before_long_request_finishes():
    events = []

    class Model:
        def __init__(self, path, device):
            self.speakers, self.languages = ['ryan'], ['english']
            self.active = {}
            self.saw_replacement = False

        def tick(self, submissions, emit, cancelled):
            for identifier in cancelled:
                self.active.pop(identifier, None)
            for identifier, payload in submissions:
                text = payload['input']
                events.append(('admit', text))
                self.active[identifier] = text
                self.saw_replacement |= text == 'replacement'
            completed = []
            for identifier, text in list(self.active.items()):
                if text != 'long' or self.saw_replacement:
                    pcm = b'\x01\x00' * 240
                    emit(identifier, pcm, 24000)
                    completed.append((identifier, {'pcm': pcm, 'sample_rate': 24000,
                                                   'codec_frames': 1, 'finish_reason': 'stop'}))
                    events.append(('finish', text))
                    del self.active[identifier]
            return completed

        def close(self):
            pass

    async def exercise():
        options = SimpleNamespace(model=Path('unused'), device='unused', queue_size=8,
                                  max_batch_size=2, batch_wait_ms=0, streaming=True,
                                  continuous=True, prefill_per_tick=2)
        service = SpeechService(options, Model)
        app = web.Application()
        app.router.add_post('/speech', service.speech)
        app.on_startup.append(service.start)
        app.on_cleanup.append(service.stop)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = f'http://127.0.0.1:{runner.addresses[0][1]}/speech'
        try:
            async with ClientSession() as session:
                async def request(text):
                    async with session.post(url, json={'input': text}) as response:
                        body = await response.text()
                        assert 'speech.audio.done' in body
                long = asyncio.create_task(request('long'))
                await request('short')
                await request('replacement')
                await long
        finally:
            await runner.cleanup()
        assert events.index(('admit', 'replacement')) < events.index(('finish', 'long'))

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_prefill_interval_batches_admission_but_cancellation_remains_immediate():
    from qwen3_tts_engine.server import AudioChannel
    events = []

    async def exercise():
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in range(3)]
        service = None
        def enqueue_replacement():
            futures[1].cancel()
            service.queue.put_nowait(({'input': 'replacement'}, futures[2], AudioChannel(loop)))
            enqueued.set()
        enqueued = threading.Event()

        class Model:
            def __init__(self, path, device):
                self.active = {}
                self.step = 0
            def tick(self, submissions, emit, cancelled):
                events.append((self.step, [p['input'] for _, p in submissions], list(cancelled)))
                for identifier in cancelled: self.active.pop(identifier, None)
                self.active.update(dict(submissions))
                if self.step == 0:
                    loop.call_soon_threadsafe(enqueue_replacement)
                    assert enqueued.wait(2)
                outputs = []
                if self.step >= 4:
                    outputs = [(identifier, {'pcm': b'\x01\x00', 'sample_rate': 24000}) for identifier in self.active]
                    self.active.clear()
                self.step += 1
                return outputs
            def close(self): pass

        options = SimpleNamespace(model=Path('unused'), device='unused', queue_size=8,
                                  max_batch_size=4, streaming=False, continuous=True,
                                  prefill_per_tick=4, prefill_interval=4)
        service = SpeechService(options, Model)
        await service.start(None)
        try:
            for i, name in enumerate(['long', 'cancel-me']):
                service.queue.put_nowait(({'input': name}, futures[i], AudioChannel(loop)))
            await asyncio.wait_for(asyncio.gather(futures[0], futures[2]), timeout=5)
            await asyncio.wait_for(service.queue.join(), timeout=2)
        finally:
            await service.stop(None)
        assert futures[1].cancelled()
        assert events[0][1] == ['long', 'cancel-me']
        assert events[1][2] == [1]
        assert all(not event[1] for event in events[1:4])
        assert events[4][1] == ['replacement']

    asyncio.run(exercise())
