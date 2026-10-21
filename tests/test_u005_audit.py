import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.audit_u005_campaign import (
    BASE,
    BASE_FLAGS,
    EXTRA_FLAGS,
    RETAINED_NOISE_FLOOR,
    REGRESSION,
    WORKLOAD,
    VIEW_PARENT_CONTINUOUS,
    VIEW_PARENT_SNAPSHOT,
    audit,
    cohort_gate,
)


def samples(values, tails=(10, 10, 10)):
    return [{"throughput_frames_s": t, "ttfa_p95_ms": p} for t, p in zip(values, tails)]


class U005AuditTests(unittest.TestCase):
    def test_missing_or_unverified_request_view_cases_cannot_pass(self):
        for field, value in (
            ("input_cases", []),
            ("generation_cases", []),
            ("views_used_for_multiple_requests", 1),
            ("immutable_random_sources_exact", False),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                path = root / "views-operator.json"
                data = json.loads(path.read_text())
                data[field] = value
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "request view"):
                    audit(root)

    def test_request_view_parent_must_match_frozen_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "views-operator.json"
            data = json.loads(path.read_text())
            data["control_continuous_sha256"] = "9" * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "parent binding"):
                audit(root)

    def test_malformed_memory_observation_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "r1-candidate-lifecycle.json"
            data = json.loads(path.read_text())
            data["hbm_after_ready"]["used_percent"] = True
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "memory observation"):
                audit(root)

    def test_untested_server_flag_combination_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "identity.json"
            data = json.loads(path.read_text())
            data["qualified_server_flags"]["candidate"].remove("--packed-kv-load")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "qualified server flags"):
                audit(root)

    def test_missing_actual_admission_state_case_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "packed-operator.json"
            data = json.loads(path.read_text())
            data["admission_state_pairs"] -= 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "admission state checks"):
                audit(root)

    def test_missing_reset_slot_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "operator.json"
            data = json.loads(path.read_text())
            data["slot_reset_cases"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "native reset checks"):
                audit(root)

    def test_missing_singleton_shape_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "operator.json"
            data = json.loads(path.read_text())
            data["singleton_cases"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "singleton Codec"):
                audit(root)

    def test_wrong_progressive_callback_order_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "operator.json"
            data = json.loads(path.read_text())
            data["progressive_callback_cases"][0]["callback_order"] = [1, 0]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "callback checks"):
                audit(root)

    def test_regression_requires_matching_source_model_and_verified_commands(self):
        for field, value in (
            ("source_snapshots_after", {"control": "9" * 64, "candidate": "8" * 64}),
            ("model_assets_after", {"weights": "7" * 64}),
            ("commands_verified", False),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                path = root / "regression.json"
                data = json.loads(path.read_text())
                data[field] = value
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "Regression identity"):
                    audit(root)

    def test_prerequisites_must_check_all_manifest_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "checks.json"
            data = json.loads(path.read_text())
            data["source_manifest_unchanged"] = False
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "different source"):
                audit(root)

    def test_primary_repeat_noise_is_inconclusive(self):
        self.assertEqual(
            cohort_gate(32, samples([100] * 3), samples([110, 117, 110]))["gate"],
            "INCONCLUSIVE",
        )

    def test_gain_inside_measured_aa_noise_is_inconclusive(self):
        self.assertEqual(
            cohort_gate(64, samples([100] * 3), samples([103] * 3), 0.04)["gate"],
            "INCONCLUSIVE",
        )
        self.assertEqual(
            cohort_gate(64, samples([100] * 3), samples([103] * 3), 0.01)["gate"],
            "PASS",
        )

    def test_aa_must_use_baseline_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "aa-second-c32.json"
            data = json.loads(path.read_text())
            data["source_revision"] = "snapshot-sha256:" + "7" * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "use baseline"):
                audit(root)

    def test_context_changes_are_disclosed_in_shared_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "r1-candidate-lifecycle.json"
            data = json.loads(path.read_text())
            data["device_monitor"]["unexpected_samples"] = 1
            path.write_text(json.dumps(data))
            self.assertEqual(audit(root)["shared_context_change_samples"], 1)

    def test_partial_formal_coverage_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "assessment.json"
            data = json.loads(path.read_text())
            data["formal_coverage"] = False
            path.write_text(json.dumps(data))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.audit_u005_campaign",
                    "--evidence",
                    folder,
                    "--output",
                    str(root / "audit.json"),
                ],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(
                json.loads((root / "audit.json").read_text())["status"], "INCONCLUSIVE"
            )

    def test_nonpython_source_changes_cannot_hide_behind_python_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "identity.json"
            data = json.loads(path.read_text())
            data["source_manifests_after"]["candidate"]["LICENSE"] = "6" * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "manifests changed"):
                audit(root)

    def test_metadata_auditor_imports_without_accelerator_packages(self):
        program = """import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'qwen_tts', 'triton'}:
        raise ImportError('accelerator dependency forbidden in control audit')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import tools.audit_u005_campaign
"""
        result = subprocess.run([sys.executable, "-c", program], capture_output=True)
        self.assertEqual(result.returncode, 0)

    def fixture(self, root):
        """Entirely synthetic records exercise the metadata auditor, not inference."""

        def save(name, data):
            (root / name).write_text(json.dumps(data), encoding="utf-8")

        sources = {
            "control": {
                "qwen3_tts_engine/codec_slots.py": "0" * 64,
                "qwen3_tts_engine/prefill_graph.py": "4" * 64,
            },
            "candidate": {
                "qwen3_tts_engine/codec_slots.py": "1" * 64,
                "qwen3_tts_engine/prefill_graph.py": "5" * 64,
            },
        }
        flags = {
            "control": list(BASE_FLAGS),
            "candidate": list(BASE_FLAGS + EXTRA_FLAGS),
        }
        memory = {
            "capacity_mb": 65536,
            "used_percent": 25,
            "scope": "synthetic driver aggregate; not an allocation peak",
        }
        snapshots = {
            arm: hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for arm, files in sources.items()
        }
        save(
            "identity.json",
            {
                "hardware": "one Ascend 910B3",
                "physical_device_index": 1,
                "base_revision": BASE,
                "workload_sha256": WORKLOAD,
                "source_before": sources,
                "source_after": sources,
                "source_snapshots": snapshots,
                "source_manifests": sources,
                "source_manifests_after": sources,
                "environment_before": {"synthetic": True},
                "environment_after": {"synthetic": True},
                "model_assets": {"weights": "3" * 64},
                "model_assets_after": {"weights": "3" * 64},
                "model_revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
                "model_revision_binding": "PASS",
                "qualified_server_flags": flags,
            },
        )
        save(
            "checks.json",
            {
                "status": "PASS",
                "tested_python_sources": sources["candidate"],
                "tested_source_snapshot": snapshots["candidate"],
                "sources_unchanged": True,
                "tested_source_manifest": sources["candidate"],
                "source_manifest_unchanged": True,
                "checks": [
                    {"check": name, "exit": code, "expected_exit": code}
                    for name, code in [
                        ("unittest", 0),
                        ("pytest", 0),
                        ("codec-npu", 0),
                        ("packed-kv-npu", 0),
                        ("request-views-npu", 0),
                        ("invalid-cli", 2),
                        ("invalid-packed-cli", 2),
                        ("invalid-views-cli", 2),
                    ]
                ],
            },
        )
        save(
            "views-operator.json",
            {
                "status": "PASS",
                "actual_process_exit": 0,
                "control_snapshot_sha256": VIEW_PARENT_SNAPSHOT,
                "control_continuous_sha256": VIEW_PARENT_CONTINUOUS,
                "shared_components_match_parent": True,
                "arena_slots": 2,
                "workload_sha256": WORKLOAD,
                "input_cases": [
                    {
                        "dtype": dtype,
                        "strided": strided,
                        "seed": seed,
                        "frame": frame,
                        "rotary_delta": {17: -13, 42: 0, 43: 17}[seed],
                        "all_inputs_consumers_and_source_exact": True,
                    }
                    for dtype in ("torch.bfloat16", "torch.float32")
                    for strided in (False, True)
                    for seed in (17, 42, 43)
                    for frame in (0, 1, 2, 7, 63, 127)
                ],
                "generation_cases": [
                    {
                        "kind": "one-two-one-cancel" if index == 3 else "single",
                        "seed": seed,
                        "ticks": 4,
                        "real_eos_count": 1,
                        "one_to_two_to_one": index == 3,
                        "default_path_pcm_and_all_state_exact": True,
                        "views_pcm_and_all_state_exact": True,
                    }
                    for index, seed in enumerate((17, 42, 43, 17))
                ],
                "borrowed_input_calls": 8,
                "views_used_for_multiple_requests": 0,
                "default_path_exact": True,
                "immutable_random_sources_exact": True,
                "actual_transition_exact": True,
            },
        )
        save(
            "packed-operator.json",
            {
                "status": "PASS",
                "actual_process_exit": 0,
                "control_prefill_sha256": sources["control"][
                    "qwen3_tts_engine/prefill_graph.py"
                ],
                "prefill_cases": [
                    {
                        "prefill_batch": batch,
                        "admitted_rows": rows,
                        "changed_replays": 3,
                        "default_path_exact": True,
                        "hidden_logits_cache_exact": True,
                        "packed_refresh_exact": True,
                        "all_arena_state_exact": True,
                        "owned_after_replay_exact": True,
                    }
                    for batch, rows in ((1, 1), (4, 1), (4, 3), (4, 4))
                ],
                "admission_state_pairs": 81,
                "admission_uniforms_and_owned_seen_exact": True,
                "immutable_initial_inputs_exact": True,
                "packed_moves_and_capacity_growth_exact": True,
                "packed_target_storage_not_duplicated": True,
                "fair_timing_input_binding": True,
            },
        )
        save(
            "operator.json",
            {
                "status": "PASS",
                "live_prewarm_exact": True,
                "range_reset_exact": True,
                "captured_decode_graphs": {"control": 32, "candidate": 64},
                "owner_lane_transition_steps": 3,
                "owner_lane_transition_exact": True,
                "slot_reset_cases": [
                    {
                        "slot": slot,
                        "changed_state_replays": 2,
                        "all_buffers_exact": True,
                    }
                    for slot in range(16)
                ],
                "singleton_callback_exception_exact": True,
                "singleton_cases": [
                    {
                        "batch": batch,
                        "length": length,
                        "changed_input_replays": 2,
                        "callback_and_all_buffers_exact": True,
                    }
                    for batch in range(1, 5)
                    for length in range(1, 9)
                ],
                "callback_exception_fence_exact": True,
                "progressive_callback_cases": [
                    {
                        "step": step,
                        "groups": groups,
                        "callback_order": list(range(groups)),
                        "pcm_exact": True,
                    }
                    for step, groups in (
                        (1, 2),
                        (3, 3),
                        (5, 2),
                        (7, 2),
                        (9, 3),
                        (11, 2),
                    )
                ],
                "control_codec_sha256": sources["control"][
                    "qwen3_tts_engine/codec_slots.py"
                ],
                "cases": [
                    {
                        "step": step,
                        "groups": groups,
                        "exact_output_and_all_buffers": True,
                    }
                    for step, groups in enumerate([2, 2, 4, 3, 2, 2] * 2)
                ],
                "copy_cases": [
                    {"rank": rank, "dtype": dtype, "exact_replays": 10}
                    for rank in (1, 2, 3, 4)
                    for dtype in ("torch.bfloat16", "torch.int64")
                ],
            },
        )
        records = [
            {
                "index": i,
                "finish_reason": "stop",
                "pcm_bytes": 3840,
                "stream_bytes": 3840,
                "audio_seconds": 0.08,
                "ttfa_seconds": 0.01,
                "elapsed_seconds": 0.1,
                "codec_frames": 1,
                "pcm_sha256": "4" * 64,
            }
            for i in range(128)
        ]
        values = {
            arm: {
                "throughput_frames_s": tp,
                "ttfa_p95_ms": 10.0,
                "wall_seconds": 128 / tp,
            }
            for arm, tp in [("control", 100.0), ("candidate", 104.0)]
        }
        monitor = {
            "samples": 100,
            "interval_seconds": 1.0,
            "max_gap_seconds": 1.2,
            "unexpected_samples": 0,
            "query_errors": 0,
            "thread_stopped": True,
        }
        save(
            "calibration.json",
            {
                "status": "PASS",
                "comparison_kind": "control-vs-control",
                "base_revision": BASE,
                "workload_sha256": WORKLOAD,
                "physical_device_index": 1,
                "source_before": sources["control"],
                "source_after": sources["control"],
                "environment_before": {"synthetic": True},
                "environment_after": {"synthetic": True},
                "model_assets": {"weights": "3" * 64},
                "baseline_commands_verified": True,
                "server_flags": list(BASE_FLAGS),
                "literal_pcm_pairs_equal": 256,
                "timed_requests": 512,
                "observed_throughput_noise_ratio": 0.0,
                "retained_noise_floor": RETAINED_NOISE_FLOOR,
            },
        )
        for arm in ("first", "second"):
            save(
                f"aa-{arm}-lifecycle.json",
                {
                    "server_exit": 0,
                    "forced_kill": False,
                    "physical_device_index": 1,
                    "device_monitor": monitor,
                    "hbm_after_ready": memory,
                    "hbm_after_cohorts": memory,
                    "cohorts": [{"concurrency": c, "exit": 0} for c in (32, 64)],
                },
            )
            for concurrency in (32, 64):
                save(
                    f"aa-{arm}-c{concurrency}.json",
                    {
                        "metrics": values["control"],
                        "requests": records,
                        "source_revision": BASE,
                        "workload_sha256": WORKLOAD,
                    },
                )
        cohorts = (1, 2, 4, 8, 16, 32, 64)
        for repeat in (1, 2, 3):
            for arm in values:
                prefix = f"r{repeat}-{arm}"
                save(
                    prefix + "-lifecycle.json",
                    {
                        "resource_policy": "opportunistic-hbm-calibrated",
                        "physical_device_index": 1,
                        "container_server_pid": 17,
                        "owned_device_pids": [12345],
                        "background_device_pids_at_ready": [],
                        "single_selected_device_process": True,
                        "selected_device_pids": [12345],
                        "hbm_after_ready": memory,
                        "hbm_after_cohorts": memory,
                        "device_monitor": {
                            "samples": 100,
                            "interval_seconds": 1.0,
                            "max_gap_seconds": 1.2,
                            "unexpected_samples": 0,
                            "query_errors": 0,
                            "thread_stopped": True,
                        },
                        "server_exit": 0,
                        "forced_kill": False,
                        "cohorts": [{"concurrency": c, "exit": 0} for c in cohorts],
                    },
                )
                for c in cohorts:
                    save(
                        prefix + f"-c{c}.json",
                        {
                            "metrics": values[arm],
                            "requests": records,
                            "source_revision": BASE
                            if arm == "control"
                            else "snapshot-sha256:" + snapshots["candidate"],
                            "workload_sha256": WORKLOAD,
                        },
                    )
        rows = []
        for c in cohorts:
            row = cohort_gate(c, [values["control"]] * 3, [values["candidate"]] * 3)
            row.update(
                control=[values["control"]] * 3, candidate=[values["candidate"]] * 3
            )
            rows.append(row)
        save(
            "assessment.json",
            {
                "formal_coverage": True,
                "paired_wavs": 2688,
                "timed_requests": 5376,
                "collector_literal_pcm_pairs_equal": 2688,
                "cohorts": rows,
                "status": "PASS",
            },
        )
        save(
            "regression.json",
            {
                "workload_sha256": REGRESSION,
                "source_before": sources,
                "source_after": sources,
                "source_snapshots_before": snapshots,
                "source_snapshots_after": snapshots,
                "environment_before": {"synthetic": True},
                "environment_after": {"synthetic": True},
                "model_assets_before": {"weights": "3" * 64},
                "model_assets_after": {"weights": "3" * 64},
                "physical_device_index": 1,
                "commands_verified": True,
                "qualified_server_flags": flags,
                "cohorts": [
                    {
                        "concurrency": c,
                        "literal_pcm_pairs_equal": 32,
                        "source_revisions": {
                            "control": BASE,
                            "candidate": "snapshot-sha256:" + snapshots["candidate"],
                        },
                        "device_monitors": {
                            arm: {
                                "samples": 100,
                                "interval_seconds": 1.0,
                                "max_gap_seconds": 1.2,
                                "unexpected_samples": 0,
                                "query_errors": 0,
                                "thread_stopped": True,
                            }
                            for arm in ("control", "candidate")
                        },
                        "server_exits": [0, 0],
                        "memory_observations": {
                            arm: {"ready": memory, "after_cohorts": memory}
                            for arm in ("control", "candidate")
                        },
                        "benchmark_exits": [0, 0],
                        "control": records[:32],
                        "candidate": records[:32],
                    }
                    for c in (4, 16)
                ],
            },
        )

    def test_complete_synthetic_metadata_and_tampered_source_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            self.assertEqual(audit(root)["status"], "PASS")
            data = json.loads((root / "checks.json").read_text())
            data["tested_python_sources"] = {"different.py": "5" * 64}
            (root / "checks.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "different source"):
                audit(root)

    def test_failed_lifecycle_cannot_be_hidden_by_passing_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "r2-candidate-lifecycle.json"
            data = json.loads(path.read_text())
            data["server_exit"] = -6
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "server exit"):
                audit(root)

    def test_geometric_mean_win_cannot_hide_strict_failure(self):
        self.assertEqual(
            cohort_gate(32, samples([100, 101, 102]), samples([106, 106, 103]))["gate"],
            "FAIL",
        )

    def test_both_primary_loads_require_more_than_two_percent(self):
        for concurrency in (32, 64):
            self.assertEqual(
                cohort_gate(concurrency, samples([100] * 3), samples([102] * 3))[
                    "gate"
                ],
                "FAIL",
            )
            self.assertEqual(
                cohort_gate(concurrency, samples([100] * 3), samples([103] * 3))[
                    "gate"
                ],
                "PASS",
            )

    def test_tail_uses_extrema_even_when_paired_tails_improve(self):
        self.assertEqual(
            cohort_gate(
                64, samples([100] * 3, [10, 11, 12]), samples([105] * 3, [9, 10, 11])
            )["gate"],
            "FAIL",
        )

    def test_nontarget_cohorts_still_have_regression_guard(self):
        self.assertEqual(
            cohort_gate(1, samples([100] * 3), samples([96] * 3))["gate"], "FAIL"
        )
        self.assertEqual(
            cohort_gate(1, samples([100] * 3), samples([97] * 3))["gate"], "PASS"
        )

    def test_invalid_measurements_do_not_pass(self):
        for bad in (float("nan"), float("inf"), 0, -1, True):
            with self.assertRaises(ValueError):
                cohort_gate(32, samples([100] * 3), samples([105, bad, 105]))

    def test_missing_record_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "audit.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.audit_u005_campaign",
                    "--evidence",
                    folder,
                    "--output",
                    str(output),
                ],
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
