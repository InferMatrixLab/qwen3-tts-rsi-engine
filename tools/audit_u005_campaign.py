"""Replay RFC U005's frozen gates from redacted records, without torch/NPU.

This audits metadata; original WAV bytes require a separate private-media audit.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.audit_fixed_rope_campaign import COHORTS, checked_cohort, read, require


BASE = "a29f8193b557c415175d70919bdfb9bd3dcd51fc"
WORKLOAD = "5007716c83b29d0247b2c6291b9704c652e01a94362c19e90ea3cb34987060c7"
REGRESSION = "5178871d85ccf659256125230958aaf7b5edd47909bbb17ac734019482acddb9"
BASE_FLAGS = tuple(
    (
        "--continuous --streaming --pack-qkv --max-batch-size 64 "
        "--codec-left-context 72 --codec-batch-size 4 --incremental-codec "
        "--pack-mlp --native-kv-update --batch-prefill --prefill-per-tick 64 "
        "--text-only-rope --prewarm-codec --native-gqa --graph-embeddings "
        "--graph-prefill --rms-norm native --fused-swiglu --native-rope "
        "--graph-sampling --native-rope-scope predictor --triton-snake "
        "--cache-codec-codebooks --triton-kv-update --triton-talker-rope "
        "--prewarm-prefill --triton-cdf --bucket-prefill --direct-prompt"
    ).split()
)
EXTRA_FLAGS = ("--codec-streams=2", "--packed-kv-load", "--single-request-views")
VIEW_PARENT_SNAPSHOT = (
    "646520ace810ee85b77fa44a513b75e6f1adb85644e885ee375be83b50736630"
)
VIEW_PARENT_CONTINUOUS = (
    "402ff616a6108b0ab78c65d39a7698b747b41db508870348bb2cf38a86675b11"
)
# Earlier same-baseline local A/A observations remain a conservative floor.
# A fresh calibration may raise this floor, never erase that prior noise.
RETAINED_NOISE_FLOOR = 0.025794542676382815


class IncompleteEvidence(RuntimeError):
    """Required verification has not completed; never convert it into PASS."""


def checked_memory(record):
    require(
        type(record["capacity_mb"]) is int
        and record["capacity_mb"] > 0
        and type(record["used_percent"]) is int
        and 0 <= record["used_percent"] <= 100,
        "Invalid driver memory observation",
    )


def checked_monitor(record):
    require(
        type(record["samples"]) is int
        and record["interval_seconds"] == 1.0
        and type(record["max_gap_seconds"]) in (int, float)
        and math.isfinite(record["max_gap_seconds"])
        and record["max_gap_seconds"] >= 0
        and type(record["unexpected_samples"]) is int
        and record["unexpected_samples"] >= 0
        and type(record["query_errors"]) is int
        and record["query_errors"] >= 0,
        "Invalid shared-device telemetry",
    )
    if not (
        record["samples"] > 0
        and record["max_gap_seconds"] <= 5
        and record["query_errors"] == 0
        and record["thread_stopped"] is True
    ):
        raise IncompleteEvidence("Incomplete shared-device telemetry")


def cohort_gate(concurrency, control, candidate, noise_floor=0.0):
    require(concurrency in COHORTS, "Unknown concurrency")
    require(len(control) == len(candidate) == 3, "Three repeats required")
    require(
        all(
            isinstance(v[k], (int, float))
            and not isinstance(v[k], bool)
            and math.isfinite(v[k])
            and v[k] > 0
            for v in control + candidate
            for k in ("throughput_frames_s", "ttfa_p95_ms")
        ),
        "Invalid performance measurement",
    )
    ratio = min(v["throughput_frames_s"] for v in candidate) / max(
        v["throughput_frames_s"] for v in control
    )
    p95 = max(v["ttfa_p95_ms"] for v in candidate) / min(
        v["ttfa_p95_ms"] for v in control
    )
    passed = (
        ratio > 1.02 if concurrency in (32, 64) else ratio >= 0.97
    ) and p95 <= 1.05
    require(
        type(noise_floor) in (int, float)
        and math.isfinite(noise_floor)
        and noise_floor >= 0,
        "Invalid calibration estimate",
    )
    spreads = {
        arm: (
            max(v["throughput_frames_s"] for v in values)
            - min(v["throughput_frames_s"] for v in values)
        )
        / sorted(v["throughput_frames_s"] for v in values)[1]
        for arm, values in (("control", control), ("candidate", candidate))
    }
    uncertain = concurrency in (32, 64) and (
        max(spreads.values()) > 0.05
        or (noise_floor > 0 and abs(ratio - 1) <= noise_floor)
    )
    return {
        "concurrency": concurrency,
        "strict_throughput_ratio": ratio,
        "conservative_p95_ratio": p95,
        "throughput_spread": spreads,
        "calibration_noise_ratio": noise_floor,
        "gate": "INCONCLUSIVE" if uncertain else "PASS" if passed else "FAIL",
    }


def regression_gate(data, identity):
    require(data["workload_sha256"] == REGRESSION, "Unqualified regression workload")
    require(
        data["source_before"] == data["source_after"] == identity["source_before"]
        and data["source_snapshots_before"]
        == data["source_snapshots_after"]
        == identity["source_snapshots"]
        and data["environment_before"]
        == data["environment_after"]
        == identity["environment_before"]
        and data["model_assets_before"]
        == data["model_assets_after"]
        == identity["model_assets"]
        and data["physical_device_index"] == identity["physical_device_index"]
        and data["commands_verified"] is True,
        "Regression identity or command binding failed",
    )
    require(
        data["qualified_server_flags"] == identity["qualified_server_flags"],
        "Regression server flags differ",
    )
    require(
        [c["concurrency"] for c in data["cohorts"]] == [4, 16],
        "Incomplete regression cohorts",
    )
    for cohort in data["cohorts"]:
        require(
            cohort["source_revisions"]
            == {
                "control": BASE,
                "candidate": "snapshot-sha256:"
                + identity["source_snapshots"]["candidate"],
            },
            "Regression source label differs",
        )
        require(
            cohort["literal_pcm_pairs_equal"] == 32, "Incomplete original-WAV check"
        )
        require(
            cohort["server_exits"] == [0, 0] and cohort["benchmark_exits"] == [0, 0],
            "Regression lifecycle failure",
        )
        require(
            set(cohort["device_monitors"]) == {"control", "candidate"},
            "Missing regression monitoring",
        )
        for monitor in cohort["device_monitors"].values():
            checked_monitor(monitor)
        require(
            set(cohort["memory_observations"]) == {"control", "candidate"},
            "Missing regression memory observations",
        )
        for memory in cohort["memory_observations"].values():
            checked_memory(memory["ready"])
            checked_memory(memory["after_cohorts"])
        arms = []
        for arm in ("control", "candidate"):
            rows = cohort[arm]
            require(
                sorted(r["index"] for r in rows) == list(range(32)),
                "Incomplete regression requests",
            )
            require(
                all(
                    r["finish_reason"] == "stop"
                    and r["pcm_bytes"] == r["stream_bytes"] > 0
                    and math.isclose(r["audio_seconds"], r["pcm_bytes"] / 48000)
                    for r in rows
                ),
                "Invalid regression audio",
            )
            arms.append(
                [(r["pcm_sha256"], r["pcm_bytes"], r["codec_frames"]) for r in rows]
            )
        require(arms[0] == arms[1], "Regression PCM differs")


def calibration_gate(root, identity):
    calibration = read(root, "calibration.json")
    if calibration["status"] != "PASS":
        raise IncompleteEvidence("Calibration not completed")
    require(
        calibration["comparison_kind"] == "control-vs-control"
        and calibration["base_revision"] == BASE
        and calibration["workload_sha256"] == WORKLOAD
        and calibration["physical_device_index"] == identity["physical_device_index"]
        and calibration["source_before"]
        == calibration["source_after"]
        == identity["source_before"]["control"]
        and calibration["environment_before"]
        == calibration["environment_after"]
        == identity["environment_before"]
        and calibration["model_assets"] == identity["model_assets"]
        and calibration["baseline_commands_verified"] is True,
        "Calibration binding failed",
    )
    require(
        calibration["literal_pcm_pairs_equal"] == 256
        and calibration["timed_requests"] == 512,
        "Incomplete original A/A PCM checks",
    )
    require(calibration["server_flags"] == list(BASE_FLAGS), "A/A server flags differ")
    observations = []
    for concurrency in (32, 64):
        pair = []
        throughput = []
        for arm in ("first", "second"):
            lifecycle = read(root, f"aa-{arm}-lifecycle.json")
            require(
                lifecycle["server_exit"] == 0
                and lifecycle["forced_kill"] is False
                and lifecycle["physical_device_index"]
                == identity["physical_device_index"]
                and lifecycle["cohorts"]
                == [{"concurrency": c, "exit": 0} for c in (32, 64)],
                "A/A lifecycle failed",
            )
            checked_monitor(lifecycle["device_monitor"])
            checked_memory(lifecycle["hbm_after_ready"])
            checked_memory(lifecycle["hbm_after_cohorts"])
            data = read(root, f"aa-{arm}-c{concurrency}.json")
            require(
                data["source_revision"] == BASE and data["workload_sha256"] == WORKLOAD,
                "A/A did not use baseline",
            )
            metrics, pcm = checked_cohort(data)
            throughput.append(metrics["throughput_frames_s"])
            pair.append(pcm)
        require(pair[0] == pair[1], "A/A PCM differs")
        observations.append(max(throughput) / min(throughput) - 1)
    noise = max(observations)
    require(
        math.isclose(
            noise,
            calibration["observed_throughput_noise_ratio"],
            rel_tol=1e-10,
            abs_tol=1e-12,
        ),
        "Calibration estimate differs",
    )
    require(
        calibration["retained_noise_floor"] == RETAINED_NOISE_FLOOR,
        "Prior noise floor changed",
    )
    return max(noise, RETAINED_NOISE_FLOOR)


def audit(root, expected_device=None):
    identity, summary, checks = (
        read(root, n) for n in ("identity.json", "assessment.json", "checks.json")
    )
    require(
        identity["qualified_server_flags"]
        == {"control": list(BASE_FLAGS), "candidate": list(BASE_FLAGS + EXTRA_FLAGS)},
        "Wrong qualified server flags",
    )
    require(
        identity["base_revision"] == BASE and identity["workload_sha256"] == WORKLOAD,
        "Unqualified baseline or corpus",
    )
    require(
        identity["source_before"] == identity["source_after"]
        and set(identity["source_before"]) == {"control", "candidate"}
        and all(identity["source_before"].values()),
        "Source changed or missing",
    )
    require(
        type(identity["physical_device_index"]) is int
        and 0 <= identity["physical_device_index"] < 8
        and (
            expected_device is None
            or identity["physical_device_index"] == expected_device
        )
        and identity["hardware"] == "one Ascend 910B3",
        "Wrong device window",
    )
    for arm in ("control", "candidate"):
        files = identity["source_manifests"][arm]
        snapshot = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        require(
            snapshot == identity["source_snapshots"][arm]
            and {name: value for name, value in files.items() if name.endswith(".py")}
            == identity["source_before"][arm],
            "Source snapshot binding failed",
        )
    require(
        identity["environment_before"] == identity["environment_after"]
        and bool(identity["environment_before"]),
        "Environment changed or missing",
    )
    require(
        identity["model_assets"] == identity["model_assets_after"]
        and bool(identity["model_assets"])
        and identity["model_revision"] == "0c0e3051f131929182e2c023b9537f8b1c68adfe"
        and identity["model_revision_binding"] == "PASS",
        "Model identity failed",
    )
    require(
        checks["tested_source_snapshot"] == identity["source_snapshots"]["candidate"],
        "Prerequisites snapshot differs",
    )
    require(
        identity["source_manifests"] == identity["source_manifests_after"],
        "Source manifests changed",
    )
    required_checks = {
        "unittest": 0,
        "pytest": 0,
        "codec-npu": 0,
        "packed-kv-npu": 0,
        "request-views-npu": 0,
        "invalid-cli": 2,
        "invalid-packed-cli": 2,
        "invalid-views-cli": 2,
    }
    require(
        checks["status"] == "PASS"
        and {c["check"] for c in checks["checks"]} == set(required_checks),
        "Incomplete prerequisites",
    )
    require(
        all(
            type(c["exit"]) is int
            and type(c["expected_exit"]) is int
            and c["exit"] == c["expected_exit"] == required_checks[c["check"]]
            for c in checks["checks"]
        ),
        "Unexpected prerequisite exit",
    )
    require(
        checks["tested_python_sources"] == identity["source_before"]["candidate"]
        and checks["sources_unchanged"] is True
        and checks["tested_source_manifest"]
        == identity["source_manifests"]["candidate"]
        and checks["source_manifest_unchanged"] is True,
        "Prerequisites apply to different source",
    )
    operator = read(root, "operator.json")
    views = read(root, "views-operator.json")
    require(
        views["status"] == "PASS"
        and type(views["actual_process_exit"]) is int
        and views["actual_process_exit"] == 0
        and views["control_snapshot_sha256"] == VIEW_PARENT_SNAPSHOT
        and views["control_continuous_sha256"] == VIEW_PARENT_CONTINUOUS
        and views["shared_components_match_parent"] is True
        and views["arena_slots"] == 2
        and views["workload_sha256"] == WORKLOAD,
        "Request views prerequisite or parent binding failed",
    )
    require(
        len(views["input_cases"]) == 72
        and {
            (r["dtype"], r["strided"], r["seed"], r["frame"])
            for r in views["input_cases"]
        }
        == {
            (dtype, strided, seed, frame)
            for dtype in ("torch.bfloat16", "torch.float32")
            for strided in (False, True)
            for seed in (17, 42, 43)
            for frame in (0, 1, 2, 7, 63, 127)
        }
        and all(
            r["all_inputs_consumers_and_source_exact"] is True
            and r["rotary_delta"] == {17: -13, 42: 0, 43: 17}[r["seed"]]
            for r in views["input_cases"]
        ),
        "Incomplete request view input cases",
    )
    require(
        len(views["generation_cases"]) == 4
        and [(r["kind"], r["seed"]) for r in views["generation_cases"]]
        == [("single", 17), ("single", 42), ("single", 43), ("one-two-one-cancel", 17)]
        and all(
            type(r["ticks"]) is int
            and r["ticks"] >= 2
            and r["real_eos_count"] == 1
            and r["default_path_pcm_and_all_state_exact"] is True
            and r["views_pcm_and_all_state_exact"] is True
            for r in views["generation_cases"]
        )
        and views["generation_cases"][-1]["one_to_two_to_one"] is True
        and type(views["borrowed_input_calls"]) is int
        and views["borrowed_input_calls"] > 0
        and type(views["views_used_for_multiple_requests"]) is int
        and views["views_used_for_multiple_requests"] == 0
        and all(
            views[key] is True
            for key in (
                "default_path_exact",
                "immutable_random_sources_exact",
                "actual_transition_exact",
            )
        ),
        "Incomplete request view actual tick checks",
    )
    packed = read(root, "packed-operator.json")
    require(
        packed["status"] == "PASS"
        and type(packed["actual_process_exit"]) is int
        and packed["actual_process_exit"] == 0
        and packed["control_prefill_sha256"]
        == identity["source_before"]["control"]["qwen3_tts_engine/prefill_graph.py"],
        "Packed KV prerequisite or baseline binding failed",
    )
    require(
        len(packed["prefill_cases"]) == 4
        and [(c["prefill_batch"], c["admitted_rows"]) for c in packed["prefill_cases"]]
        == [(1, 1), (4, 1), (4, 3), (4, 4)]
        and all(
            type(c["prefill_batch"]) is int
            and type(c["admitted_rows"]) is int
            and c["changed_replays"] == 3
            and all(
                c[key] is True
                for key in (
                    "default_path_exact",
                    "hidden_logits_cache_exact",
                    "packed_refresh_exact",
                    "all_arena_state_exact",
                    "owned_after_replay_exact",
                )
            )
            for c in packed["prefill_cases"]
        ),
        "Incomplete packed KV cases",
    )
    require(
        packed["admission_state_pairs"] == 81
        and all(
            packed[key] is True
            for key in (
                "admission_uniforms_and_owned_seen_exact",
                "immutable_initial_inputs_exact",
                "packed_moves_and_capacity_growth_exact",
                "packed_target_storage_not_duplicated",
                "fair_timing_input_binding",
            )
        ),
        "Incomplete admission state checks",
    )
    require(
        len(operator["slot_reset_cases"]) == 16
        and {c["slot"] for c in operator["slot_reset_cases"]} == set(range(16))
        and all(
            c["changed_state_replays"] == 2 and c["all_buffers_exact"] is True
            for c in operator["slot_reset_cases"]
        ),
        "Incomplete or failed native reset checks",
    )
    require(
        operator["captured_decode_graphs"] == {"control": 32, "candidate": 64},
        "Wrong Codec capture set",
    )
    require(
        operator["owner_lane_transition_steps"] == 3
        and operator["owner_lane_transition_exact"] is True,
        "Incomplete owner/lane transition checks",
    )
    require(
        operator["singleton_callback_exception_exact"] is True
        and len(operator["singleton_cases"]) == 32
        and {(c["batch"], c["length"]) for c in operator["singleton_cases"]}
        == {(batch, length) for batch in range(1, 5) for length in range(1, 9)}
        and all(
            c["changed_input_replays"] == 2
            and c["callback_and_all_buffers_exact"] is True
            for c in operator["singleton_cases"]
        ),
        "Incomplete or failed singleton Codec checks",
    )
    require(
        operator["status"] == "PASS"
        and operator["live_prewarm_exact"] is True
        and operator["range_reset_exact"] is True,
        "Failed or missing Codec state checks",
    )
    require(
        operator["callback_exception_fence_exact"] is True
        and [(c["step"], c["groups"]) for c in operator["progressive_callback_cases"]]
        == [(1, 2), (3, 3), (5, 2), (7, 2), (9, 3), (11, 2)]
        and all(
            c["callback_order"] == list(range(c["groups"])) and c["pcm_exact"] is True
            for c in operator["progressive_callback_cases"]
        ),
        "Incomplete or failed progressive callback checks",
    )
    require(
        [(c["step"], c["groups"]) for c in operator["cases"]]
        == list(enumerate([2, 2, 4, 3, 2, 2] * 2))
        and all(c["exact_output_and_all_buffers"] is True for c in operator["cases"]),
        "Incomplete or failed Codec cases",
    )
    require(
        {(c["rank"], c["dtype"], c["exact_replays"]) for c in operator["copy_cases"]}
        == {
            (rank, dtype, 10)
            for rank in (1, 2, 3, 4)
            for dtype in ("torch.bfloat16", "torch.int64")
        }
        and len(operator["copy_cases"]) == 8
        and operator["control_codec_sha256"]
        == identity["source_before"]["control"]["qwen3_tts_engine/codec_slots.py"],
        "Incomplete copy checks or wrong baseline Codec",
    )
    if summary["formal_coverage"] is not True:
        raise IncompleteEvidence("Formal coverage incomplete")
    require(
        summary["formal_coverage"] is True
        and summary["paired_wavs"] == 2688
        and summary["timed_requests"] == 5376
        and summary["collector_literal_pcm_pairs_equal"] == 2688,
        "Incomplete formal coverage",
    )
    require(
        [r["concurrency"] for r in summary["cohorts"]] == list(COHORTS),
        "Missing summary cohorts",
    )
    noise_floor = calibration_gate(root, identity)
    rows = []
    for concurrency in COHORTS:
        values = {"control": [], "candidate": []}
        for repeat in (1, 2, 3):
            pair = []
            for arm in values:
                prefix = f"r{repeat}-{arm}"
                lifecycle = read(root, prefix + "-lifecycle.json")
                checked_monitor(lifecycle["device_monitor"])
                checked_memory(lifecycle["hbm_after_ready"])
                checked_memory(lifecycle["hbm_after_cohorts"])
                require(
                    type(lifecycle["container_server_pid"]) is int
                    and lifecycle["container_server_pid"] > 0
                    and lifecycle["physical_device_index"]
                    == identity["physical_device_index"]
                    and lifecycle["resource_policy"] == "opportunistic-hbm-calibrated",
                    "Device ownership missing",
                )
                require(
                    lifecycle["server_exit"] == 0 and lifecycle["forced_kill"] is False,
                    "Unclean server exit",
                )
                require(
                    lifecycle["cohorts"]
                    == [{"concurrency": c, "exit": 0} for c in COHORTS],
                    "Missing or failed benchmark process",
                )
                data = read(root, prefix + f"-c{concurrency}.json")
                expected = (
                    BASE
                    if arm == "control"
                    else "snapshot-sha256:" + identity["source_snapshots"][arm]
                )
                require(
                    data["source_revision"] == expected
                    and data["workload_sha256"] == WORKLOAD,
                    "Wrong source or workload label",
                )
                metrics, pcm = checked_cohort(data)
                values[arm].append(metrics)
                pair.append(pcm)
            require(pair[0] == pair[1], "PCM differs")
        row = cohort_gate(
            concurrency, values["control"], values["candidate"], noise_floor
        )
        recorded = next(
            r for r in summary["cohorts"] if r["concurrency"] == concurrency
        )
        require(
            all(recorded[arm] == values[arm] for arm in values),
            "Summary measurements differ",
        )
        require(
            row["gate"] == recorded["gate"]
            and all(
                math.isclose(row[k], recorded[k], rel_tol=1e-10)
                for k in ("strict_throughput_ratio", "conservative_p95_ratio")
            ),
            "Recorded gate differs",
        )
        rows.append(row)
    regression_gate(read(root, "regression.json"), identity)
    status = (
        "FAIL"
        if any(r["gate"] == "FAIL" for r in rows)
        else "INCONCLUSIVE"
        if any(r["gate"] == "INCONCLUSIVE" for r in rows)
        else "PASS"
    )
    require(status == summary["status"], "Recorded overall verdict differs")
    return {
        "status": status,
        "shared_context_change_samples": sum(
            read(root, f"r{repeat}-{arm}-lifecycle.json")["device_monitor"][
                "unexpected_samples"
            ]
            for repeat in (1, 2, 3)
            for arm in ("control", "candidate")
        ),
        "calibration_noise_ratio": noise_floor,
        "memory_observations": {
            f"r{repeat}-{arm}": {
                key: read(root, f"r{repeat}-{arm}-lifecycle.json")[key]
                for key in ("hbm_after_ready", "hbm_after_cohorts")
            }
            for repeat in (1, 2, 3)
            for arm in ("control", "candidate")
        },
        "cohorts": rows,
        "timed_requests": 5376,
        "paired_requests": 2688,
        "regression_pairs": 64,
        "scope": "recorded metadata only; no fresh NPU run or independent WAV reads",
        "evidence_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.json"))
            if not p.name.startswith("audit-")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, choices=range(8))
    args = parser.parse_args()
    try:
        report = audit(args.evidence, args.device_index)
    except (OSError, KeyError, IncompleteEvidence) as error:
        report = {"status": "INCONCLUSIVE", "error_type": type(error).__name__}
    except (ValueError, TypeError, ArithmeticError) as error:
        report = {"status": "FAIL", "error_type": type(error).__name__}
    try:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(
            json.dumps({"status": "INCONCLUSIVE", "error_type": type(error).__name__})
        )
        return 2
    print(json.dumps({"status": report["status"], "scope": "metadata only"}))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
