"""Independent-arena Codec stream experiment; no service qualification."""

import argparse
import json
from pathlib import Path
import time

import torch

from qwen3_tts_engine.codec_slots import CodecSlotGraph, CodecSlots
from qwen3_tts_engine.model import CustomVoiceModel


class StreamGraph(CodecSlotGraph):
    """Capture and replay on the same explicit stream, with private graph storage."""

    def capture_on(self, stream):
        saved = [buffer.clone() for buffer in self.arena.buffers]
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.output = self.compute()
        torch.npu.synchronize()
        for buffer, original in zip(self.arena.buffers, saved):
            buffer.copy_(original)
        torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "status": "INCONCLUSIVE",
        "scope": "independent-arena component upper bound only",
        "cases": [],
        "timings": [],
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    arenas = []
    model = None
    try:
        with torch.inference_mode():
            model = CustomVoiceModel(args.model, "npu:0")
            decoder = model.wrapper.model.speech_tokenizer.model.decoder
            arenas = [
                CodecSlots(decoder, 8, triton_snake=True, cache_codebooks=True)
                for _ in range(4)
            ]
            serial_stream = torch.npu.Stream()
            streams = [
                serial_stream,
                serial_stream,
                torch.npu.Stream(),
                torch.npu.Stream(),
            ]
            for index, (arena, stream) in enumerate(zip(arenas, streams)):
                for length in (2, 8):
                    graph = StreamGraph(arena, 4, length)
                    arena.graphs[4, length] = graph
                    graph.capture_on(stream)
                result["captures_completed"] = index + 1
                save()

            torch.manual_seed(149)
            previous = {}
            for step in range(12):
                length = 2 if step % 2 == 0 else 8
                indices = [0, 1, 2, 3] if step < 6 else [4, 1, 2, 3]
                if step == 6:
                    for arena in arenas:
                        arena.move(0, 4)
                        arena.reset(0)
                if step == 9:
                    for arena in arenas:
                        arena.reset(1)
                codes = [
                    torch.randint(0, 2048, (4, 16, length), device="npu:0")
                    for _ in range(2)
                ]
                producer = torch.npu.current_stream()
                for stream in set(streams):
                    stream.wait_stream(producer)
                outputs = []
                for index, arena in enumerate(arenas):
                    with torch.npu.stream(streams[index]):
                        outputs.append(
                            arena.graphs[4, length].run(indices, codes[index % 2])
                        )
                for stream in set(streams):
                    producer.wait_stream(stream)
                torch.npu.synchronize()
                for group in (0, 1):
                    if not torch.equal(outputs[group], outputs[group + 2]):
                        raise AssertionError("waveform mismatch")
                    if not all(
                        torch.equal(x, y)
                        for x, y in zip(
                            arenas[group].buffers, arenas[group + 2].buffers
                        )
                    ):
                        raise AssertionError("state mismatch")
                if length in previous:
                    if torch.equal(previous[length], outputs[0]):
                        raise AssertionError("changed input did not change output")
                previous[length] = outputs[0].clone()
                result["cases"].append({"step": step, "length": length, "exact": True})
                save()

            # No timings are collected until all changed-input/state checks pass.
            for length in (2, 8):
                for arm in ("serial", "parallel", "parallel", "serial"):
                    selected = (0, 1) if arm == "serial" else (2, 3)
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(100):
                        for index in selected:
                            with torch.npu.stream(streams[index]):
                                graph = arenas[index].graphs[4, length]
                                graph.graph.replay()
                                graph.output.clone()
                    torch.npu.synchronize()
                    result["timings"].append(
                        {
                            "length": length,
                            "arm": arm,
                            "pair_ms": (time.perf_counter() - start) * 10,
                        }
                    )
                    save()
            result["status"] = "PASS"
    except AssertionError:
        result.update(status="FAIL", failure_kind="exactness_or_input_responsiveness")
    except Exception as error:
        result.update(status="INCONCLUSIVE", error_type=type(error).__name__)
    finally:
        for arena in arenas:
            try:
                arena.close()
            except Exception as error:
                result.update(
                    status="INCONCLUSIVE", cleanup_error_type=type(error).__name__
                )
        if model is not None:
            try:
                model.close()
            except Exception as error:
                result.update(
                    status="INCONCLUSIVE", cleanup_error_type=type(error).__name__
                )
        save()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
