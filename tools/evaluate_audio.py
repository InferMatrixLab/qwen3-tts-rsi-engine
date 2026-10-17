"""Independently transcribe retained PCM with CPU Whisper and score WER/CER."""

import argparse
import hashlib
import json
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--asr-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
parser.add_argument("--limit", type=int, help="Validator qualification subset; not full-corpus evidence")
options = parser.parse_args()
torch.set_num_threads(4)
if options.device.startswith("npu"):
    import torch_npu
    torch.npu.set_device(options.device)
dtype = getattr(torch, options.dtype)
run = json.loads((options.run / "results.json").read_text())
spec = json.loads((options.run / "runspec.json").read_text())
workload = Path(spec["workload"]).read_bytes()
if hashlib.sha256(workload).hexdigest() != spec["workload_sha256"]:
    raise ValueError("Workload changed after measurement")
payloads = [json.loads(line) for line in workload.splitlines() if line.strip()]
processor = WhisperProcessor.from_pretrained(options.asr_model, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    options.asr_model, local_files_only=True, dtype=dtype,
).to(options.device).eval()
options.output.mkdir(parents=True, exist_ok=False)
english = EnglishTextNormalizer()
basic = BasicTextNormalizer()
rows = []
records = run["requests"] if options.limit is None else run["requests"][:options.limit]
for record in records:
    index = record["index"]
    samples, rate = sf.read(options.run / f"{index:05d}.wav", dtype="int16")
    if not record["success"] or samples.ndim != 1 or not len(samples):
        raise ValueError(f"Invalid request audio: {index}")
    if hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest() != record["pcm_sha256"]:
        raise ValueError(f"Audio changed after measurement: {index}")
    waveform = torchaudio.functional.resample(torch.from_numpy(samples.astype(np.float32) / 32768), rate, 16000)
    locale = "zh" if payloads[index]["language"].lower() == "chinese" else "en"
    segments = []
    for offset in range(0, len(waveform), 30 * 16000):
        inputs = processor(waveform[offset:offset + 30 * 16000].numpy(), sampling_rate=16000,
                           return_tensors="pt", return_attention_mask=True)
        inputs = {key: tensor.to(device=options.device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
                  for key, tensor in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(**inputs, language=locale, task="transcribe", max_new_tokens=440)
        segments.append(processor.batch_decode(tokens, skip_special_tokens=True)[0])
    transcript = " ".join(segments)
    reference = payloads[index]["input"]
    if locale == "en":
        normalized_reference, normalized_transcript = english(reference), english(transcript)
        errors = jiwer.process_words(normalized_reference, normalized_transcript)
        metric = "WER"
    else:
        normalized_reference = "".join(basic(reference).split())
        normalized_transcript = "".join(basic(transcript).split())
        errors = jiwer.process_characters(normalized_reference, normalized_transcript)
        metric = "CER"
    row = {"index": index, "locale": locale, "reference": reference, "transcript": transcript,
           "normalized_reference": normalized_reference, "normalized_transcript": normalized_transcript,
           "metric": metric, "substitutions": errors.substitutions, "deletions": errors.deletions,
           "insertions": errors.insertions, "hits": errors.hits, "segments": len(segments),
           "pcm_sha256": record["pcm_sha256"]}
    rows.append(row)
    with (options.output / "samples.jsonl").open("a") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Scored {len(rows)}/{len(run['requests'])}", flush=True)
metrics = {}
for locale in ("en", "zh"):
    selected = [r for r in rows if r["locale"] == locale]
    count = sum(r["hits"] + r["substitutions"] + r["deletions"] for r in selected)
    if count:
        metrics[locale] = {"metric": selected[0]["metric"], "samples": len(selected),
                           "error_rate": sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in selected) / count}
summary = {"status": "INCONCLUSIVE", "scored": len(rows), "expected": len(run["requests"]),
           "metrics": metrics, "asr_model": str(options.asr_model), "asr_device": options.device,
           "precision": options.dtype, "speaker_similarity": "NOT_RUN",
           "workload_sha256": spec["workload_sha256"],
           "notes": "Observed ASR quality; matched baseline quality and preset speaker similarity required."}
(options.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
