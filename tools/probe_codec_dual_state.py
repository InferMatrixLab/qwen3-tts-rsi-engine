"""Exact shared-arena dual-stream Codec checks against an external baseline file."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from qwen3_tts_engine.codec_row_scatter import scatter_rows
from qwen3_tts_engine.codec_slots import CodecSlots
from qwen3_tts_engine.model import CustomVoiceModel


def check_scatter():
    cases = []
    for dtype in (torch.bfloat16, torch.long):
        for inner in ((), (17,), (3, 19), (2, 9, 7)):
            target = torch.zeros((8,) + inner, dtype=dtype, device="npu:0")
            indices = torch.tensor([5, 1, 7], device="npu:0")
            # KV tails and sliced histories are not necessarily contiguous.
            source = torch.ones((3,) + inner + (2,), dtype=dtype, device="npu:0")[
                ..., 0
            ]
            scatter_rows(target, indices, source)
            expected = target.clone()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            try:
                with torch.npu.graph(graph, stream=stream):
                    scatter_rows(target, indices, source)
                for replay in range(10):
                    source.fill_(replay + 3)
                    expected.index_copy_(0, indices, source)
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        graph.replay()
                    torch.npu.current_stream().wait_stream(stream)
                    if not torch.equal(target, expected):
                        raise AssertionError("pure copy mismatch")
            finally:
                torch.npu.synchronize()
                graph.reset()
            cases.append(
                {"rank": target.ndim, "dtype": str(dtype), "exact_replays": 10}
            )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "INCONCLUSIVE", "scope": "component only", "cases": []}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    engines = []
    model = None
    try:
        with torch.inference_mode():
            torch.npu.set_device("npu:0")
            result["copy_cases"] = check_scatter()
            save()
            spec = importlib.util.spec_from_file_location(
                "control_codec", args.control_codec
            )
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            result["control_codec_sha256"] = hashlib.sha256(
                args.control_codec.read_bytes()
            ).hexdigest()
            model = CustomVoiceModel(args.model, "npu:0")
            decoder = model.wrapper.model.speech_tokenizer.model.decoder
            control = baseline.CodecSlots(
                decoder, 16, triton_snake=True, cache_codebooks=True
            )
            candidate = CodecSlots(
                decoder, 16, triton_snake=True, cache_codebooks=True, streams=2
            )
            engines.extend((control, candidate))
            torch.manual_seed(181)
            result["singleton_cases"] = []
            for batch in range(1, 5):
                for length in range(1, 9):
                    slots = [11, 3, 9, 6][:batch]
                    for replay in range(2):
                        codes = torch.randint(
                            0, 2048, (batch, 16, length), device="npu:0"
                        )
                        expected = control.decode(slots, codes)
                        delivered = []
                        actual = candidate.decode_groups(
                            [(slots, codes)],
                            on_output=lambda index, wave: delivered.append(
                                (index, wave.float().cpu())
                            ),
                        )[0]
                        if (
                            not torch.equal(expected, actual)
                            or len(delivered) != 1
                            or delivered[0][0] != 0
                            or not torch.equal(expected.float().cpu(), delivered[0][1])
                            or not all(
                                torch.equal(x, y)
                                for x, y in zip(control.buffers, candidate.buffers)
                            )
                        ):
                            raise AssertionError("singleton output or state mismatch")
                    result["singleton_cases"].append(
                        {
                            "batch": batch,
                            "length": length,
                            "changed_input_replays": 2,
                            "callback_and_all_buffers_exact": True,
                        }
                    )
                    save()
            previous = {}
            schedule = [
                [([0, 1, 2, 3], 2), ([4, 5, 6, 7], 2)],
                [([0, 1, 2, 3], 8), ([4, 5, 6, 7], 8)],
                [([4, 5, 6, 7], 8), ([0, 1, 2, 3], 8), ([8, 9], 2), ([10], 1)],
                [([10], 7), ([0, 1, 2], 2), ([4, 5, 6, 7], 8)],
                [([11, 1, 2, 3], 8), ([4, 5, 6, 7], 8)],
                [([11, 1, 2, 3], 8), ([4, 5, 6, 7], 8)],
            ]
            for step, groups in enumerate(schedule * 2):
                if step == 8:
                    for slot in (4, 5):
                        control.reset(slot)
                    candidate.reset_range(4, 2)
                    if not all(
                        torch.equal(x, y)
                        for x, y in zip(control.buffers, candidate.buffers)
                    ):
                        raise AssertionError("range reset mismatch")
                    result["range_reset_exact"] = True
                if step in (4, 10):
                    for engine in engines:
                        engine.move(0, 11)
                        engine.reset(0)
                        engine.reset(5)
                codes = [
                    torch.randint(0, 2048, (len(slots), 16, length), device="npu:0")
                    for slots, length in groups
                ]
                expected = [
                    control.decode(slots, code)
                    for (slots, _), code in zip(groups, codes)
                ]
                delivered = []

                def consume(index, waveform):
                    delivered.append((index, waveform.float().cpu()))

                actual = candidate.decode_groups(
                    [(slots, code) for (slots, _), code in zip(groups, codes)],
                    on_output=consume if step % 2 else None,
                )
                torch.npu.synchronize()
                if len(actual) != len(expected) or len(actual) != len(groups):
                    raise AssertionError("missing group output")
                if not all(torch.equal(x, y) for x, y in zip(expected, actual)):
                    raise AssertionError("waveform mismatch")
                if step % 2:
                    if [index for index, _ in delivered] != list(range(len(groups))):
                        raise AssertionError("callback count or order mismatch")
                    if not all(
                        torch.equal(expected[index].float().cpu(), wave)
                        for index, wave in delivered
                    ):
                        raise AssertionError("callback waveform mismatch")
                    result.setdefault("progressive_callback_cases", []).append(
                        {
                            "step": step,
                            "groups": len(groups),
                            "callback_order": [index for index, _ in delivered],
                            "pcm_exact": True,
                        }
                    )
                if not all(
                    torch.equal(x, y)
                    for x, y in zip(control.buffers, candidate.buffers)
                ):
                    raise AssertionError("state mismatch")
                shape = tuple(actual[0].shape)
                if shape in previous and torch.equal(previous[shape], actual[0]):
                    raise AssertionError("unchanged output")
                previous[shape] = actual[0].clone()
                result["cases"].append(
                    {
                        "step": step,
                        "groups": len(groups),
                        "exact_output_and_all_buffers": True,
                    }
                )
                save()

            class CallbackStopped(RuntimeError):
                pass

            def stop_delivery(*_):
                raise CallbackStopped()

            # Ensure a callback exception still fences later queued groups before
            # owner-stream state copies. A global sync before these copies would
            # hide a missing dependency, so enqueue the snapshots first.
            groups = [([10], 1), ([0, 1, 2, 3], 8), ([4, 5, 6, 7], 8)]
            work = [
                (
                    slots,
                    torch.randint(0, 2048, (len(slots), 16, length), device="npu:0"),
                )
                for slots, length in groups
            ]
            for slots, codes in work:
                control.decode(slots, codes)
            try:
                candidate.decode_groups(work, on_output=stop_delivery)
            except CallbackStopped:
                pass
            else:
                raise AssertionError("missing callback exception")
            snapshots = [buffer.clone() for buffer in candidate.buffers]
            torch.npu.synchronize()
            if not all(torch.equal(x, y) for x, y in zip(control.buffers, snapshots)):
                raise AssertionError("callback exception state fence mismatch")
            result["callback_exception_fence_exact"] = True
            codes = torch.randint(0, 2048, (2, 16, 8), device="npu:0")
            control.decode([3, 9], codes)
            try:
                candidate.decode_groups([([3, 9], codes)], on_output=stop_delivery)
            except CallbackStopped:
                pass
            else:
                raise AssertionError("missing singleton callback exception")
            snapshots = [buffer.clone() for buffer in candidate.buffers]
            torch.npu.synchronize()
            if not all(torch.equal(x, y) for x, y in zip(control.buffers, snapshots)):
                raise AssertionError("singleton callback exception state mismatch")
            result["singleton_callback_exception_exact"] = True
            save()
            for engine in engines:
                saved = [buffer.clone() for buffer in engine.buffers]
                engine.prewarm(4)
                torch.npu.synchronize()
                if not all(torch.equal(x, y) for x, y in zip(saved, engine.buffers)):
                    raise AssertionError("prewarm changed live state")
            result["live_prewarm_exact"] = True
            result["captured_decode_graphs"] = {
                "control": len(control.graphs),
                "candidate": len(candidate.graphs),
            }
            transitions = [
                [([0, 1, 2, 3], 8), ([4, 5, 6, 7], 8)],
                [([0, 1], 2)],
                [([0, 1], 2), ([4, 5, 6, 7], 8)],
            ]
            work = [
                [
                    (
                        slots,
                        torch.randint(
                            0, 2048, (len(slots), 16, length), device="npu:0"
                        ),
                    )
                    for slots, length in groups
                ]
                for groups in transitions
            ]
            single_indices = torch.tensor(work[1][0][0], device="npu:0")
            expected = [
                [control.decode(slots, codes) for slots, codes in groups]
                for groups in work
            ]
            actual = []
            for groups in work:
                # Prebuilt device indices avoid hiding a missing lane fence
                # behind an incidental host-to-device index transfer.
                if len(groups) == 1:
                    slots, codes = groups[0]
                    actual.append(
                        [
                            candidate._graph(len(slots), codes.shape[-1]).run(
                                single_indices, codes
                            )
                        ]
                    )
                else:
                    actual.append(candidate.decode_groups(groups))
            snapshots = [buffer.clone() for buffer in candidate.buffers]
            if not all(
                torch.equal(x, y)
                for xs, ys in zip(expected, actual)
                for x, y in zip(xs, ys)
            ) or not all(torch.equal(x, y) for x, y in zip(control.buffers, snapshots)):
                raise AssertionError("owner/lane transition mismatch")
            result["owner_lane_transition_steps"] = 3
            result["owner_lane_transition_exact"] = True
            result["slot_reset_cases"] = []
            for slot in range(candidate.slots):
                for replay in range(2):
                    for engine in engines:
                        for buffer in engine.buffers:
                            buffer[slot].fill_(replay + 1)
                    control.reset(slot)
                    candidate.reset(slot)
                    snapshots = [buffer.clone() for buffer in candidate.buffers]
                    torch.npu.synchronize()
                    if not all(
                        torch.equal(x, y) for x, y in zip(control.buffers, snapshots)
                    ):
                        raise AssertionError(
                            "native reset changed selected or unrelated state"
                        )
                result["slot_reset_cases"].append(
                    {
                        "slot": slot,
                        "changed_state_replays": 2,
                        "all_buffers_exact": True,
                    }
                )
                save()
            result["timings"] = []
            for length in (2, 8):
                codes = [
                    torch.randint(0, 2048, (4, 16, length), device="npu:0")
                    for _ in range(2)
                ]
                groups = [([0, 1, 2, 3], codes[0]), ([4, 5, 6, 7], codes[1])]
                for arm in ("control", "candidate", "candidate", "control"):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(80):
                        if arm == "control":
                            for slots, code in groups:
                                control.decode(slots, code)
                        else:
                            candidate.decode_groups(groups)
                    torch.npu.synchronize()
                    result["timings"].append(
                        {
                            "length": length,
                            "arm": arm,
                            "pair_ms": (time.perf_counter() - start) * 12.5,
                        }
                    )
                    save()
            result["status"] = "PASS"
    except AssertionError:
        result.update(status="FAIL", failure_kind="exactness_or_input_responsiveness")
    except Exception as error:
        result.update(status="INCONCLUSIVE", error_type=type(error).__name__)
    finally:
        for engine in engines:
            try:
                engine.close()
            except Exception as error:
                result.update(cleanup_error_type=type(error).__name__)
                if result["status"] != "FAIL":
                    result["status"] = "INCONCLUSIVE"
        if model is not None:
            try:
                model.close()
            except Exception as error:
                result.update(cleanup_error_type=type(error).__name__)
                if result["status"] != "FAIL":
                    result["status"] = "INCONCLUSIVE"
        save()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
