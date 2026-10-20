"""Recompute U002 gates from redacted evidence (stdlib only, no accelerator).

This checks recorded metadata. The A2 collector separately opens the original
WAV files; those private originals are required to independently verify audio.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


COHORTS = (1, 2, 4, 8, 16, 32, 64)


def read(root, name):
    return json.loads((root / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cohort_gate(concurrency, control, candidate):
    require(len(control) == len(candidate) == 3, "Three paired repeats required")
    ratios = [b["throughput_frames_s"] / a["throughput_frames_s"]
              for a, b in zip(control, candidate)]
    ratio = statistics.geometric_mean(ratios)
    p95 = max(v["ttfa_p95_ms"] for v in candidate) / min(v["ttfa_p95_ms"] for v in control)
    passed = ratio >= .97 and p95 <= 1.05
    if concurrency == 1:
        passed = passed and ratio > 1.01 and min(ratios) > 1
    return {"concurrency": concurrency, "paired_throughput_ratios": ratios,
            "geomean_throughput_ratio": ratio, "conservative_p95_ratio": p95,
            "gate": "PASS" if passed else "FAIL"}


def checked_cohort(data):
    rows = data["requests"]
    require(sorted(row["index"] for row in rows) == list(range(128)), "Incomplete request coverage")
    for row in rows:
        require(row["finish_reason"] == "stop", "Request did not reach EOS")
        require(row["pcm_bytes"] == row["stream_bytes"] > 0, "Stream byte mismatch")
        require(math.isclose(row["audio_seconds"], row["pcm_bytes"] / 48000), "PCM duration mismatch")
        require(len(row["pcm_sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in row["pcm_sha256"]), "Invalid PCM hash")
        require(all(math.isfinite(row[k]) and row[k] > 0
                    for k in ("audio_seconds", "ttfa_seconds", "elapsed_seconds", "codec_frames")),
                "Invalid request measurement")
    metrics = data["metrics"]
    require(math.isfinite(metrics["wall_seconds"]) and metrics["wall_seconds"] > 0, "Invalid wall time")
    throughput = sum(row["audio_seconds"] for row in rows) * 12.5 / metrics["wall_seconds"]
    values = sorted(row["ttfa_seconds"] * 1000 for row in rows)
    index = (len(values) - 1) * .95
    lo = int(index)
    p95 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
    require(math.isclose(throughput, metrics["throughput_frames_s"], rel_tol=1e-8), "Throughput mismatch")
    require(math.isclose(p95, metrics["ttfa_p95_ms"], rel_tol=1e-8), "p95 mismatch")
    return metrics, [(r["pcm_sha256"], r["pcm_bytes"], r["codec_frames"]) for r in rows]


def audit(root):
    checks = read(root, "checks.json")
    require(checks["status"] == "PASS" and all(v == 0 for v in checks["exits"].values())
            and set(checks["exits"]) == {"UNIT_EXIT", "PYTEST_EXIT", "OPERATOR_EXIT"}, "Prerequisite failure")
    final = read(root, "final-checks.json")
    require(final["status"] == "PASS" and final["measured_python_sources_unchanged"],
            "Final validation failed")
    expected_exits = {"syntax": 0, "unittest": 0, "pytest": 0, "package-import": 0, "cli-invalid": 2}
    require(set(final["checks"]) == set(expected_exits), "Incomplete final checks")
    require(all(final["checks"][name]["status"] == "PASS"
                and final["checks"][name]["exit"] == value for name, value in expected_exits.items()),
            "Unexpected final check exit")
    operator = read(root, "operator.json")
    require(operator["status"] == "PASS" and len(operator["cases"]) == 40, "Incomplete predictor checks")
    expected_cases = {(batch, replay, embeddings) for batch in (1, 4, 16, 32, 64)
                      for replay in range(4) for embeddings in (False, True)}
    require({(r["batch"], r["replay"], r["graph_embeddings"]) for r in operator["cases"]}
            == expected_cases, "Missing predictor case")
    require(all(len(r["outputs_exact"]) == (2 if r["graph_embeddings"] else 1)
                and all(r["outputs_exact"]) for r in operator["cases"]), "Predictor outputs differ")
    identity = read(root, "identity.json")
    require(identity["source_before"] == identity["source_after"], "Source changed")
    require(set(identity["source_before"]) == {"control", "candidate"}
            and all(identity["source_before"].values()), "Missing source manifests")
    require(identity["model_assets"] == identity["model_assets_after"], "Model assets changed")
    require(identity["base_revision"] == "9f190577aa2a147c6840560a56c5996a6c4879a4"
            and identity["model_revision"] == "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            "Unqualified source/model revision")
    require(identity["model_revision_binding"] == "PASS", "Unverified model revision")
    summary = read(root, "assessment.json")
    require(summary["source_unchanged"] and summary["model_unchanged"], "Identity check failed")
    require(sorted(r["concurrency"] for r in summary["cohorts"]) == list(COHORTS),
            "Incomplete summary cohorts")
    result = []
    for concurrency in COHORTS:
        control, candidate = [], []
        for repeat in (1, 2, 3):
            pair = []
            for arm, metrics in (("control", control), ("candidate", candidate)):
                lifecycle = read(root, f"r{repeat}-{arm}-lifecycle.json")
                require(lifecycle["server_exit"] == 0 and not lifecycle["forced_kill"], "Unclean server exit")
                require(lifecycle["cohorts"] == [{"concurrency": c, "exit": 0} for c in COHORTS],
                        "Incomplete benchmark exits")
                data = read(root, f"r{repeat}-{arm}-c{concurrency}.json")
                expected_source = ("snapshot-sha256:" + identity["source_snapshots"]["candidate"]
                                   if arm == "candidate" else identity["base_revision"])
                require(data["source_revision"] == expected_source, "Incorrect source label")
                require(data["workload_sha256"] == identity["workload_sha256"]
                        == "5007716c83b29d0247b2c6291b9704c652e01a94362c19e90ea3cb34987060c7",
                        "Incorrect workload")
                measured, pcm = checked_cohort(data)
                metrics.append(measured)
                pair.append(pcm)
            require(pair[0] == pair[1], "PCM differs")
        row = cohort_gate(concurrency, control, candidate)
        recorded = next(r for r in summary["cohorts"] if r["concurrency"] == concurrency)
        require(recorded["control"] == control and recorded["candidate"] == candidate,
                "Summary measurements differ from request evidence")
        require(recorded["paired_throughput_ratios"] == row["paired_throughput_ratios"],
                "Summary paired ratios differ")
        require(row["gate"] == recorded["gate"], "Recorded gate mismatch")
        for name in ("geomean_throughput_ratio", "conservative_p95_ratio"):
            require(math.isclose(row[name], recorded[name], rel_tol=1e-10), "Recorded ratio mismatch")
        result.append(row)
    status = "PASS" if all(row["gate"] == "PASS" for row in result) else "FAIL"
    require(status == summary["status"], "Overall status mismatch")
    require(summary["accuracy"] == "PASS_EXACT_PCM" and summary["paired_wavs"] == 2688
            and summary["timed_requests"] == 5376
            and summary["collector_literal_pcm_pairs_equal"] == 2688, "Incorrect coverage claim")
    return {"status": status, "accuracy": "PASS_RECORDED_PCM_IDENTITIES", "cohorts": result,
            "paired_requests": 2688, "timed_requests": 5376,
            "scope": "metadata recomputation; no new NPU run or independent original-WAV inspection",
            "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.glob("*.json")) if not p.name.startswith("audit-")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.evidence)
    except (OSError, KeyError) as error:
        report = {"status": "INCONCLUSIVE", "error_type": type(error).__name__}
    except (ValueError, TypeError, ArithmeticError) as error:
        report = {"status": "FAIL", "error_type": type(error).__name__}
    try:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    except OSError as error:
        print(json.dumps({"status": "INCONCLUSIVE", "error_type": type(error).__name__,
                          "scope": "report could not be written; path details omitted"}))
        return 1
    print(json.dumps({"status": report["status"], "scope": "recorded metadata only"}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
