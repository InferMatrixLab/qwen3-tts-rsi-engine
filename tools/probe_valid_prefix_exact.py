"""Strict native FIA prefix parity; no service or speech-quality claim."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "INCONCLUSIVE",
        "scope": "synthetic exact native-operator graph comparison only",
        "cases": [],
    }
    try:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output)
    except OSError as error:
        print(
            json.dumps({"status": "INCONCLUSIVE", "error_type": type(error).__name__})
        )
        return 2
    captures = []
    try:
        torch.npu.set_device("npu:0")
        torch.manual_seed(111)
        with torch.inference_mode():
            for batch in (1, 8, 17, 32, 64):
                for length in (64, 128, 256, 512):
                    q = torch.randn(
                        batch, 16, 1, 128, device="npu:0", dtype=torch.bfloat16
                    )
                    k = torch.randn(batch, 8, 512, 128, device=q.device, dtype=q.dtype)
                    v = torch.randn_like(k)
                    mask = torch.zeros(
                        batch, 1, 1, 512, device=q.device, dtype=torch.bool
                    )

                    def operation(prefix):
                        return torch_npu.npu_fused_infer_attention_score(
                            q,
                            k,
                            v,
                            atten_mask=mask,
                            actual_seq_lengths_kv=[length] * batch if prefix else None,
                            num_heads=16,
                            num_key_value_heads=8,
                            input_layout="BNSD",
                            scale=128**-0.5,
                            sparse_mode=0,
                        )[0]

                    graphs = []
                    outputs = []
                    streams = []
                    row = {
                        "batch": batch,
                        "prefix_length": length,
                        "capacity": 512,
                        "exact_replays": 0,
                    }
                    result["cases"].append(row)
                    capture = {
                        "graphs": graphs,
                        "outputs": outputs,
                        "inputs": (q, k, v, mask),
                        "streams": streams,
                        "row": row,
                    }
                    captures.append(capture)
                    for prefix in (False, True):
                        stream = torch.npu.Stream()
                        streams.append(stream)
                        stream.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(stream):
                            for _ in range(3):
                                operation(prefix)
                        torch.npu.current_stream().wait_stream(stream)
                        torch.npu.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, stream=stream):
                            output = operation(prefix)
                        graphs.append(graph)
                        outputs.append(output)
                    for replay in range(10):
                        q.copy_(torch.randn_like(q))
                        k.copy_(torch.randn_like(k))
                        v.copy_(torch.randn_like(v))
                        positions = (
                            torch.arange(batch, device=q.device) * 17 + replay * 13
                        ) % length
                        if batch == 1:
                            positions.fill_(0 if replay % 2 == 0 else length - 1)
                        else:
                            positions[0] = 0
                            positions[-1] = length - 1
                        mask.copy_(
                            (
                                torch.arange(512, device=q.device)[None, :]
                                > positions[:, None]
                            )[:, None, None, :]
                        )
                        k[:, :, length:].fill_(100 * (replay + 1))
                        v[:, :, length:].fill_(-100 * (replay + 1))
                        for graph in graphs:
                            graph.replay()
                        expected = operation(False)
                        torch.npu.synchronize()
                        if not torch.equal(expected, outputs[0]) or not torch.equal(
                            expected, outputs[1]
                        ):
                            artifact = args.output.with_suffix(".mismatch.pt")
                            torch.save(
                                {
                                    "query": q.cpu(),
                                    "key": k.cpu(),
                                    "value": v.cpu(),
                                    "mask": mask.cpu(),
                                    "direct_reference": expected.cpu(),
                                    "full_graph": outputs[0].cpu(),
                                    "prefix_graph": outputs[1].cpu(),
                                    "prefix_length": length,
                                    "replay": replay,
                                },
                                artifact,
                            )
                            row.update(
                                failed_replay=replay,
                                max_abs_difference=float(
                                    (expected - outputs[1]).abs().max().item()
                                ),
                                different_elements=int(
                                    (expected != outputs[1]).sum().item()
                                ),
                            )
                            result["private_mismatch_artifact_sha256"] = hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest()
                            result["failure_kind"] = "exact_numerical_mismatch"
                            raise AssertionError("Exact native FIA outputs differ")
                        row["exact_replays"] += 1
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
            # No timings are collected for a candidate that fails any parity group.
            for capture in captures:
                timings = []
                for arm in (0, 1, 1, 0):
                    graph = capture["graphs"][arm]
                    for _ in range(10):
                        graph.replay()
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    for _ in range(200):
                        graph.replay()
                    torch.npu.synchronize()
                    timings.append(
                        {
                            "arm": "full" if arm == 0 else "prefix",
                            "mean_ms": (time.perf_counter() - started) * 5,
                        }
                    )
                capture["row"]["timings"] = timings
            result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__)
    finally:
        cleanup_errors = []
        for capture in captures:
            for graph in capture["graphs"]:
                try:
                    graph.reset()
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)
        if cleanup_errors:
            result["cleanup_error_types"] = cleanup_errors
            if result["status"] == "PASS":
                result["status"] = "INCONCLUSIVE"
        try:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        except OSError as error:
            print(
                json.dumps(
                    {"status": "INCONCLUSIVE", "error_type": type(error).__name__}
                )
            )
            return 2
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
