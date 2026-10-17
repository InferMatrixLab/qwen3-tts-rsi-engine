"""Own one server process group, retain requests, and record its actual exit."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--command-json", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--workload", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--engine", required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--requests", type=int, default=8)
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--concurrency-list", type=int, nargs="+")
parser.add_argument("--startup-timeout", type=float, default=900)
options = parser.parse_args()
command = json.loads(options.command_json.read_text())
if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
    parser.error("Command must be a JSON array of strings")
options.output.mkdir(parents=True, exist_ok=False)
status = {"status": "INCONCLUSIVE", "phase": "startup", "command": command,
          "engine": options.engine, "source_revision": options.source_revision,
          "quality": "NOT_RUN", "server_exit": None, "benchmark_exit": None}


def save():
    temporary = options.output / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(options.output / "status.json")


save()
with (options.output / "server.log").open("w") as log:
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status["server_pid"] = server.pid
    save()
    try:
        deadline = time.monotonic() + options.startup_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{options.port}/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server readiness timeout")
            time.sleep(2)
        status["cohorts"] = []
        for concurrency in options.concurrency_list or [options.concurrency]:
            status["phase"] = f"requests-c{concurrency}"
            destination = options.output / (f"requests-c{concurrency}" if options.concurrency_list else "requests")
            benchmark = [sys.executable, "-m", "qwen3_tts_engine.benchmark",
                         "--endpoint", f"http://127.0.0.1:{options.port}/v1/audio/speech",
                         "--model", options.model, "--engine", options.engine,
                         "--source-revision", options.source_revision,
                         "--workload", str(options.workload), "--output", str(destination),
                         "--requests", str(options.requests), "--concurrency", str(concurrency),
                         "--frame-rate", "12.5", "--warmup", str(max(2, concurrency))]
            status["benchmark_command"] = benchmark
            save()
            log_name = f"benchmark-c{concurrency}.log" if options.concurrency_list else "benchmark.log"
            with (options.output / log_name).open("w") as benchmark_log:
                result = subprocess.run(benchmark, stdout=benchmark_log, stderr=subprocess.STDOUT, timeout=7200)
            status["benchmark_exit"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"Benchmark c{concurrency} failed: {result.returncode}")
            metrics = json.loads((destination / "results.json").read_text())["metrics"]
            status["cohorts"].append({"concurrency": concurrency, "metrics": metrics})
            save()
    except Exception as error:
        status["status"] = "FAIL"
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        status["phase"] = "shutdown"
        save()
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
                status["forced_kill"] = True
        status["server_exit"] = server.returncode
        status["clean_server_exit"] = server.returncode == 0
        status["phase"] = "finished"
        save()
tail = (options.output / "server.log").read_text(errors="replace").splitlines()[-60:]
(options.output / "server-tail.json").write_text(json.dumps(tail, indent=2) + "\n")
print(json.dumps(status, indent=2))
raise SystemExit(0 if status["benchmark_exit"] == 0 and status["clean_server_exit"] else 1)
