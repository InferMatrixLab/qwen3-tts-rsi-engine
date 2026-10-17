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
