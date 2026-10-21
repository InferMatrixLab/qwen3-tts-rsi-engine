"""Exact configured-model checks for packed KV and initial semantic sampling."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from qwen3_tts_engine.continuous import ContinuousModel, SlotArena
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.prefill_graph import PrefillGraph
from qwen3_tts_engine.sampling_graph import SamplingGraph


MODEL_OPTIONS = dict(
    rms_norm="native",
    pack_qkv=True,
    pack_mlp=True,
    fused_swiglu=True,
    native_gqa=True,
    text_only_rope=True,
    native_rope=True,
    native_rope_scope="predictor",
    triton_talker_rope=True,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-prefill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "INCONCLUSIVE",
        "scope": "component only; no service qualification",
        "prefill_cases": [],
        "admission_state_pairs": 0,
        "timings": [],
        "model_options": MODEL_OPTIONS,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    graphs = []
    timing_plans = []
    model = sampler = None
    previous = None
    try:
        with torch.inference_mode():
            spec = importlib.util.spec_from_file_location(
                "u005_control_prefill", args.control_prefill
            )
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            result["control_prefill_sha256"] = hashlib.sha256(
                args.control_prefill.read_bytes()
            ).hexdigest()
            model = CustomVoiceModel(args.model, "npu:0", **MODEL_OPTIONS)
            talker = model.wrapper.model.talker
            control_arena = SlotArena(talker.model, 8, 256)
            packed_arena = SlotArena(talker.model, 8, 256, packed_kv=True)
            engine = object.__new__(ContinuousModel)
            engine.talker, engine.device = talker, "npu:0"
            engine.eos, engine.triton_cdf = talker.config.codec_eos_token_id, True
            engine.allowed = torch.ones(
                talker.config.vocab_size, device="npu:0", dtype=torch.bool
            )
            engine.allowed[-1024:] = False
            engine.allowed[engine.eos] = True
            engine.initial_seen = torch.zeros(
                1, talker.config.vocab_size, device="npu:0", dtype=torch.bool
            )
            engine.initial_steps = torch.zeros(1, device="npu:0", dtype=torch.long)
            engine.incremental_codec = False
            engine.states = []
            sampler = SamplingGraph(
                engine.semantic, 1, talker.config.vocab_size, "npu:0"
            )
            sampler.capture()
            engine.samplers = {1: sampler}
            for batch in (1, 4):
                control = baseline.PrefillGraph(talker, batch, 128)
                default = PrefillGraph(talker, batch, 128)
                packed = PrefillGraph(talker, batch, 128, packed_kv=True)
                graphs.extend((control, default, packed))
                for graph in (control, default, packed):
                    graph.capture()
                torch.npu.synchronize()
                lengths = [113] if batch == 1 else [31, 79, 113, 127]
                mask = torch.ones_like(control.mask)
                for row, length in enumerate(lengths):
                    mask[row, length:] = 0
                plans = (
                    [(1, 0, lengths)]
                    if batch == 1
                    else [(1, 0, lengths[:1]), (2, 1, lengths[1:]), (3, 0, lengths)]
                )
                for start, source_start, prefixes in plans:
                    for replay in range(3):
                        embeddings = torch.randn_like(control.inputs) * 0.1
                        hidden, logits, cache = control.run(embeddings, mask)
                        dh, dl, dc = default.run(embeddings, mask)
                        ph, pl, pc = packed.run(embeddings, mask)
                        sources = [x.keys for x in cache.layers] + [
                            x.values for x in cache.layers
                        ]
                        default_sources = [x.keys for x in dc.layers] + [
                            x.values for x in dc.layers
                        ]
                        packed_sources = [x.keys for x in pc.layers] + [
                            x.values for x in pc.layers
                        ]
                        assert torch.equal(hidden, dh) and torch.equal(hidden, ph)
                        assert torch.equal(logits, dl) and torch.equal(logits, pl)
                        assert default.packed_cache is None
                        assert all(
                            torch.equal(x, y)
                            and torch.equal(x, z)
                            and torch.equal(x, packed.packed_cache[i])
                            for i, (x, y, z) in enumerate(
                                zip(sources, default_sources, packed_sources)
                            )
                        )
                        for buffer in control_arena.keys + control_arena.values:
                            buffer.fill_(replay + 3)
                        packed_arena.packed.fill_(replay + 3)
                        for offset, length in enumerate(prefixes):
                            control_arena.load(
                                start + offset,
                                cache,
                                row=source_start + offset,
                                length=length,
                            )
                            packed_arena.load_packed(
                                start + offset,
                                packed.packed_cache,
                                row=source_start + offset,
                                length=length,
                            )
                            prompt = {
                                "trailing_text_hidden": torch.zeros_like(hidden[:1]),
                                "tts_pad_embed": torch.zeros_like(hidden[:1]),
                            }
                            for seed in (17, 42, 43):
                                payload = {"seed": seed, "max_new_tokens": 2048}
                                row = source_start + offset
                                engine.graph_sampling = False
                                engine.add_state(
                                    "synthetic-control",
                                    payload,
                                    prompt,
                                    hidden[row : row + 1],
                                    logits[row : row + 1],
                                    length,
                                    0,
                                )
                                eager = engine.states.pop()
                                sampler.run(
                                    logits[row : row + 1],
                                    eager.uniforms[0, 0:1],
                                    torch.ones_like(engine.initial_seen),
                                    engine.initial_steps + 3,
                                )
                                engine.graph_sampling = True
                                engine.add_state(
                                    "synthetic-candidate",
                                    payload,
                                    prompt,
                                    ph[row : row + 1],
                                    pl[row : row + 1],
                                    length,
                                    0,
                                )
                                actual = engine.states.pop()
                                assert torch.equal(
                                    eager.token, actual.token
                                ) and torch.equal(eager.seen, actual.seen)
                                assert torch.equal(eager.uniforms, actual.uniforms)
                                assert not bool(engine.initial_seen.any()) and not bool(
                                    engine.initial_steps.any()
                                )
                                if previous is not None:
                                    old, token, seen = previous
                                    assert torch.equal(
                                        old.token, token
                                    ) and torch.equal(old.seen, seen)
                                previous = (
                                    actual,
                                    actual.token.clone(),
                                    actual.seen.clone(),
                                )
                                result["admission_state_pairs"] += 1
                        snapshot = packed_arena.packed.clone()
                        assert all(
                            torch.equal(x, snapshot[i])
                            for i, x in enumerate(
                                control_arena.keys + control_arena.values
                            )
                        )
                        packed.run(torch.randn_like(embeddings) * 0.1, mask)
                        assert torch.equal(snapshot, packed_arena.packed)
                    result["prefill_cases"].append(
                        {
                            "prefill_batch": batch,
                            "admitted_rows": len(prefixes),
                            "changed_replays": 3,
                            "default_path_exact": True,
                            "hidden_logits_cache_exact": True,
                            "packed_refresh_exact": True,
                            "all_arena_state_exact": True,
                            "owned_after_replay_exact": True,
                        }
                    )
                    save()
                timing_plans.append(
                    (
                        batch,
                        control,
                        packed,
                        torch.randn_like(control.inputs) * 0.1,
                        mask.clone(),
                        lengths,
                    )
                )
            control_arena.move(3, 7)
            packed_arena.move(3, 7)
            assert all(
                torch.equal(x, y)
                for x, y in zip(
                    control_arena.keys + control_arena.values,
                    packed_arena.keys + packed_arena.values,
                )
            )
            assert all(
                x.is_contiguous() for x in packed_arena.keys + packed_arena.values
            )
            assert packed_arena.packed.numel() == sum(
                x.numel() for x in packed_arena.keys + packed_arena.values
            )
            engine.graphs, engine.max_slots, engine.arena = {}, 8, packed_arena
            engine.ensure_capacity(256)
            assert engine.arena.packed_kv and engine.arena.capacity == 512
            assert all(
                torch.equal(old, new[:, :, :256]) and not bool(new[:, :, 256:].any())
                for old, new in zip(
                    control_arena.keys + control_arena.values,
                    engine.arena.keys + engine.arena.values,
                )
            )
            result.update(
                admission_uniforms_and_owned_seen_exact=True,
                immutable_initial_inputs_exact=True,
                packed_moves_and_capacity_growth_exact=True,
                packed_target_storage_not_duplicated=True,
                fair_timing_input_binding=True,
            )
            # All exactness checks above must pass before any component timings.
            # Both arms copy identical independent, owned inputs into their graphs.
            for (
                batch,
                control,
                packed,
                timing_inputs,
                timing_mask,
                lengths,
            ) in timing_plans:
                for arm in ("control", "packed", "packed", "control"):
                    torch.npu.synchronize()
                    begin = time.perf_counter()
                    for _ in range(20):
                        if arm == "control":
                            _, _, cache = control.run(timing_inputs, timing_mask)
                            for row, length in enumerate(lengths):
                                control_arena.load(row, cache, row=row, length=length)
                        else:
                            packed.run(timing_inputs, timing_mask)
                            for row, length in enumerate(lengths):
                                packed_arena.load_packed(
                                    row, packed.packed_cache, row=row, length=length
                                )
                    torch.npu.synchronize()
                    result["timings"].append(
                        {
                            "prefill_batch": batch,
                            "arm": arm,
                            "prefill_and_load_wall_ms": (time.perf_counter() - begin)
                            * 50,
                        }
                    )
                    save()
            result["status"] = "PASS"
    except AssertionError:
        result.update(status="FAIL", failure_kind="exact_model_or_state")
    except Exception as error:
        result.update(status="INCONCLUSIVE", error_type=type(error).__name__)
    finally:
        for item in (
            ([sampler] if sampler is not None else [])
            + graphs
            + ([model] if model is not None else [])
        ):
            try:
                item.close()
            except Exception as error:
                result.update(cleanup_error_type=type(error).__name__)
                if result["status"] != "FAIL":
                    result["status"] = "INCONCLUSIVE"
        save()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
