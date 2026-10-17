import asyncio
import base64
import json

from aiohttp import ClientSession, web
import pytest

from qwen3_tts_engine.benchmark import request, summarize


def test_metrics_use_audio_duration_and_request_rtf():
    rows = [
        {"success": True, "audio_seconds": 2, "elapsed_seconds": 1, "ttfa_seconds": .1},
        {"success": True, "audio_seconds": 4, "elapsed_seconds": 4, "ttfa_seconds": .3},
    ]
    result = summarize(rows, 5, 12.5)
    assert result["throughput_frames_s"] == 15
    assert result["rtf_mean"] == .75
    assert result["ttfa_p50_ms"] == 200
    assert result["ttfa_p95_ms"] == 290
    assert result["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("terminal,success", [
    ({"type": "speech.audio.done", "finish_reason": "stop"}, True),
    ({"type": "speech.audio.done", "finish_reason": "length"}, False),
    ({"type": "error", "error": {"message": "worker failed"}}, False),
    (None, False),
])
def test_sse_completion_contract(tmp_path, terminal, success):
    async def exercise():
        async def handler(incoming):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(incoming)
            event = {"type": "speech.audio.delta", "audio": base64.b64encode(b"\x01\x00" * 240).decode()}
            for item in [event, terminal]:
                if item:
                    await response.write(("data: " + json.dumps(item) + "\n\n").encode())
            await response.write_eof()
            return response
        app = web.Application()
        app.router.add_post("/speech", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        server = web.TCPSite(runner, "127.0.0.1", 0)
        await server.start()
        port = runner.addresses[0][1]
        try:
            async with ClientSession() as session:
                result = await request(session, f"http://127.0.0.1:{port}/speech", {"input": "test"}, tmp_path, 0, 24000)
            assert result["success"] is success
            assert result["pcm_bytes"] == 480
            assert (tmp_path / "00000.wav").exists()
        finally:
            await runner.cleanup()
    asyncio.run(exercise())
