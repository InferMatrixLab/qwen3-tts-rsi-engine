"""Check request views against a frozen parent and actual tick state machines."""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from qwen3_tts_engine.codec_slots import CodecSlots
from qwen3_tts_engine.continuous import ContinuousModel, SlotArena
from qwen3_tts_engine import request_views
from qwen3_tts_engine.sampling import random_table
from tools.probe_packed_kv import MODEL_OPTIONS


PARENT = "646520ace810ee85b77fa44a513b75e6f1adb85644e885ee375be83b50736630"
PARENT_CONTINUOUS = "402ff616a6108b0ab78c65d39a7698b747b41db508870348bb2cf38a86675b11"


def legacy_inputs(state):
    frame = state.frames
    text = (
        state.trailing[:, frame : frame + 1]
        if frame < state.trailing.shape[1]
        else state.pad
    )
    return (
        torch.cat([state.token]),
        torch.cat([state.hidden]),
        torch.stack([state.uniforms[frame]]),
        torch.cat([text]),
        torch.stack([state.uniforms[frame + 1, 0]]),
        torch.cat([state.seen]),
        torch.tensor([state.position], device=state.token.device),
        torch.tensor([state.position + state.rope_delta], device=state.token.device)
        .view(1, 1, 1)
        .expand(3, -1, -1),
        torch.tensor([frame + 1], device=state.token.device),
    )


def consume(inputs, targets):
    for source, target in zip(inputs, targets):
        target.copy_(source)
    return targets[1] + targets[3]


def check_inputs(hidden, vocab, result, save):
    plans = []
    for dtype in (torch.bfloat16, torch.float32):
        for strided in (False, True):
            factor = 2 if strided else 1
            for seed in (17, 42, 43):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                storage = torch.zeros(129, 16 * factor, device="npu:0")
                storage[:, ::factor].copy_(
                    torch.rand(129, 16, generator=generator).to("npu:0")
                )
                state = SimpleNamespace(
                    token=torch.tensor([[seed]], device="npu:0"),
                    hidden=torch.randn(
                        1, 1, hidden * factor, device="npu:0", dtype=dtype
                    )[..., ::factor],
                    uniforms=storage[:, ::factor],
                    trailing=torch.randn(
                        1, 3, hidden * factor, device="npu:0", dtype=dtype
                    )[..., ::factor],
                    pad=torch.randn(1, 1, hidden * factor, device="npu:0", dtype=dtype)[
                        ..., ::factor
                    ],
                    seen=torch.zeros(
                        1, vocab * factor, device="npu:0", dtype=torch.bool
                    )[:, ::factor],
                    position=121 + seed,
                    rope_delta={17: -13, 42: 0, 43: 17}[seed],
                )
                state.seen[:, ::7] = True
                originals = {
                    name: getattr(state, name).clone()
                    for name in (
                        "token",
                        "hidden",
                        "uniforms",
                        "trailing",
                        "pad",
                        "seen",
                    )
                }
                for frame in (0, 1, 2, 7, 63, 127):
                    state.frames = frame
                    expected, actual = (
                        legacy_inputs(state),
                        request_views.single_request_inputs(state),
                    )
                    assert all(
                        x.shape == y.shape and x.dtype == y.dtype and torch.equal(x, y)
                        for x, y in zip(expected, actual)
                    )
                    left = [
                        torch.empty(x.shape, device=x.device, dtype=x.dtype)
                        for x in expected
                    ]
                    right = [torch.empty_like(x) for x in left]
                    assert torch.equal(consume(expected, left), consume(actual, right))
                    assert all(torch.equal(x, y) for x, y in zip(left, right))
                    assert all(
                        torch.equal(getattr(state, name), value)
                        for name, value in originals.items()
                    )
                    result["input_cases"].append(
                        {
                            "dtype": str(dtype),
                            "strided": strided,
                            "seed": seed,
                            "frame": frame,
                            "rotary_delta": state.rope_delta,
                            "all_inputs_consumers_and_source_exact": True,
                        }
                    )
                    save()
                if dtype == torch.bfloat16 and not strided and seed == 42:
                    plans.append((state, originals))
    return plans


def check_state(reference, other):
    assert len(reference.states) == len(other.states)
    for left, right in zip(reference.states, other.states):
        for name in (
            "identifier",
            "payload",
            "position",
            "rope_delta",
            "frames",
            "emitted",
            "audio",
        ):
            assert getattr(left, name) == getattr(right, name)
        for name in ("hidden", "token", "trailing", "pad", "uniforms", "seen"):
            assert torch.equal(getattr(left, name), getattr(right, name))
        assert len(left.codes) == len(right.codes)
        assert all(torch.equal(x, y) for x, y in zip(left.codes, right.codes))
    assert reference.arena.capacity == other.arena.capacity
    assert torch.equal(reference.arena.packed, other.arena.packed)
    assert len(reference.codec.buffers) == len(other.codec.buffers)
    assert all(
        torch.equal(x, y) for x, y in zip(reference.codec.buffers, other.codec.buffers)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-continuous", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "INCONCLUSIVE",
        "scope": "component and actual tick correctness; not service performance qualification",
        "arena_slots": 2,
        "input_cases": [],
        "generation_cases": [],
        "timings": [],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    control = None
    variants = []
    original_views = request_views.single_request_inputs
    active = {"engine": None, "calls": 0, "multiple_calls": 0}
    try:
        with torch.inference_mode():
            torch.npu.set_device("npu:0")
            parent_root = args.control_continuous.parent.parent
            parent_manifest = json.loads(
                (parent_root / "source-manifest.json").read_text()
            )
            assert parent_manifest["snapshot_sha256"] == PARENT
            assert (
                hashlib.sha256(args.control_continuous.read_bytes()).hexdigest()
                == PARENT_CONTINUOUS
            )
            for name, digest in parent_manifest["files"].items():
                assert (
                    hashlib.sha256((parent_root / name).read_bytes()).hexdigest()
                    == digest
                )
                if (
                    name.startswith("qwen3_tts_engine/")
                    and name.endswith(".py")
                    and name
                    not in {
                        "qwen3_tts_engine/continuous.py",
                        "qwen3_tts_engine/server.py",
                    }
                ):
                    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            result.update(
                control_snapshot_sha256=PARENT,
                control_continuous_sha256=PARENT_CONTINUOUS,
                shared_components_match_parent=True,
                workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            )
            assert (
                inspect.signature(ContinuousModel)
                .parameters["single_request_views"]
                .default
                is False
            )
            spec = importlib.util.spec_from_file_location(
                "u005_parent_continuous", args.control_continuous
            )
            baseline = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = baseline
            spec.loader.exec_module(baseline)
            control = baseline.ContinuousModel(
                args.model,
                "npu:0",
                max_slots=2,
                initial_capacity=512,
                **MODEL_OPTIONS,
                codec_left_context=72,
                codec_batch_size=4,
                incremental_codec=True,
                batch_prefill=True,
                native_kv_update=True,
                graph_prefill=True,
                prewarm_codec=True,
                graph_embeddings=True,
                bucket_prefill=True,
                graph_sampling=True,
                cache_codec_codebooks=True,
                prewarm_prefill=True,
                triton_kv_update=True,
                triton_cdf=True,
                direct_prompt=True,
                triton_snake=True,
                codec_streams=2,
                packed_kv_load=True,
            )
            decoder = control.model.wrapper.model.speech_tokenizer.model.decoder
            for enabled in (False, True):
                engine = object.__new__(ContinuousModel)
                engine.__dict__ = control.__dict__.copy()
                engine.single_request_views = enabled
                engine.arena = SlotArena(
                    control.talker.model, 2, 512, True, True, packed_kv=True
                )
                engine.graphs, engine.states = {}, []
                engine.codec = CodecSlots(
                    decoder, 2, triton_snake=True, cache_codebooks=True, streams=2
                )
                variants.append(engine)
                engine.codec.prewarm(2)
            engines = [control] + variants
            plans = check_inputs(
                control.talker.config.hidden_size,
                control.talker.config.vocab_size,
                result,
                save,
            )

            def observed_views(state):
                engine = active["engine"]
                assert engine is variants[1] and engine.single_request_views is True
                if len(engine.states) != 1:
                    active["multiple_calls"] += 1
                    raise AssertionError("views used for multiple requests")
                active["calls"] += 1
                return original_views(state)

            request_views.single_request_inputs = observed_views
            rows = [
                json.loads(line)
                for line in args.workload.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(rows) >= 3

            def payload(index, seed):
                row = rows[index]
                return {
                    "input": row["input"],
                    "voice": row["voice"],
                    "language": row["language"],
                    "instructions": row.get("instructions", ""),
                    "seed": seed,
                    "max_new_tokens": row["max_new_tokens"],
                }

            def generation(index, seed, transition=False):
                callbacks = [[] for _ in engines]
                random_sources = [{} for _ in engines]
                total_eos = 0
                saw_two = saw_one_after_two = False
                for step in range(2048):
                    submissions = [(0, payload(index, seed))] if step == 0 else []
                    if transition and step == 1:
                        submissions = [(1, payload((index + 1) % len(rows), 43))]
                    cancelled = [0] if transition and step == 2 else []
                    outputs = []
                    for arm, engine in enumerate(engines):
                        active["engine"] = engine
                        outputs.append(
                            engine.tick(
                                submissions,
                                lambda identifier, pcm, rate, arm=arm: callbacks[
                                    arm
                                ].append((identifier, pcm, rate)),
                                cancelled,
                            )
                        )
                        for request in engine.states:
                            if request.identifier not in random_sources[arm]:
                                random_sources[arm][request.identifier] = random_table(
                                    request.payload["seed"],
                                    request.payload["max_new_tokens"],
                                ).to("npu:0")
                            assert torch.equal(
                                request.uniforms,
                                random_sources[arm][request.identifier],
                            )
                    assert outputs[0] == outputs[1] == outputs[2]
                    assert callbacks[0] == callbacks[1] == callbacks[2]
                    for engine in variants:
                        check_state(control, engine)
                    saw_two |= len(control.states) == 2
                    saw_one_after_two |= saw_two and len(control.states) == 1
                    for _, output in outputs[0]:
                        assert (
                            output["finish_reason"] == "stop"
                            and output["pcm"]
                            and output["sample_rate"] == 24000
                        )
                        total_eos += 1
                    if not control.states:
                        break
                else:
                    raise AssertionError("generation did not finish")
                assert total_eos == 1
                if transition:
                    assert saw_two and saw_one_after_two
                result["generation_cases"].append(
                    {
                        "kind": "one-two-one-cancel" if transition else "single",
                        "seed": seed,
                        "ticks": step + 1,
                        "real_eos_count": total_eos,
                        "one_to_two_to_one": saw_two and saw_one_after_two,
                        "default_path_pcm_and_all_state_exact": True,
                        "views_pcm_and_all_state_exact": True,
                    }
                )
                save()

            for index, seed in enumerate((17, 42, 43)):
                generation(index, seed)
            generation(0, 17, transition=True)
            request_views.single_request_inputs = original_views
            result.update(
                default_path_exact=True,
                immutable_random_sources_exact=True,
                borrowed_input_calls=active["calls"],
                views_used_for_multiple_requests=active["multiple_calls"],
                actual_transition_exact=True,
            )
            # Correctness above, including actual graph consumption, precedes timing.
            for state, originals in plans:
                for frame in (0, 63):
                    state.frames = frame
                    targets = [
                        torch.empty(x.shape, device=x.device, dtype=x.dtype)
                        for x in legacy_inputs(state)
                    ]
                    for arm in ("copied", "views", "views", "copied"):
                        prepare = legacy_inputs if arm == "copied" else original_views
                        torch.npu.synchronize()
                        begin = time.perf_counter()
                        for _ in range(200):
                            consume(prepare(state), targets)
                        torch.npu.synchronize()
                        result["timings"].append(
                            {
                                "frame": frame,
                                "arm": arm,
                                "prepare_and_consume_ms": (time.perf_counter() - begin)
                                * 5,
                            }
                        )
                        save()
                    assert all(
                        torch.equal(x, y) for x, y in zip(targets, legacy_inputs(state))
                    )
                    assert all(
                        torch.equal(getattr(state, name), value)
                        for name, value in originals.items()
                    )
            result["status"] = "PASS"
    except AssertionError:
        result.update(status="FAIL", failure_kind="exactness_state_or_source_binding")
    except Exception as error:
        result.update(status="INCONCLUSIVE", error_type=type(error).__name__)
    finally:
        request_views.single_request_inputs = original_views
        for engine in variants:
            for item in list(engine.graphs.values()) + [engine.codec]:
                try:
                    item.close()
                except Exception as error:
                    result["cleanup_error_type"] = type(error).__name__
                    if result["status"] != "FAIL":
                        result["status"] = "INCONCLUSIVE"
        if control is not None:
            try:
                control.close()
            except Exception as error:
                result["cleanup_error_type"] = type(error).__name__
                if result["status"] != "FAIL":
                    result["status"] = "INCONCLUSIVE"
        save()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
