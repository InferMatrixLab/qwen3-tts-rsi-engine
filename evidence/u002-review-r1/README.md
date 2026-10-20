# U002 review R1 validation

Restored A2 functional/regression validation: **PASS** for `d22a97981976dde533c61ff63043acf3f09cfdf6`. See `validation-restored.json`, actual A2 checks, predictor cases, service records and source manifests. All 90 measured Python files match that commit and the evidence follow-up working tree. This follow-up changes documentation/evidence only.

The first attempt failed before synchronization because SSH timed out. Its original INCONCLUSIVE `validation.json` and local results remain unchanged. Restored results are new records, not a rewrite of that attempt.

## Review corrections

- Opt-in `UniformPredictor` construction resolves `reference.model.rotary_emb` and rejects missing/wrapped structures, training and non-default RoPE before service startup completes. The shared validator inspects metadata only. Capture validates again; disabled mode does not inspect the wrapper.
- Recommend enabling only at C1–C2. At C≥4 expect no useful throughput benefit: all original higher-load rows had two negative repeats and conservative p95 TTFA regressed up to 4.30% at C16 and 3.96% at C32. Default remains disabled.
- Tests cover constructor success without coefficient computation/capture, incompatible structures/configuration, default-off compatibility, capture-time revalidation and service startup failure before ready/worker creation.

## Actual restored A2 execution

- `python -m unittest discover -s tests -v`: **18 passed**, including all six startup integration cases previously skipped locally; per-case PASS records are in `startup-cases-restored.json`.
- `python -m pytest tests -q`: **75 passed**, 14 existing warnings, 17 passing subtests. Actual exit0 for both suites.
- Real-model predictor graph probe: **40 matched cases**, five batch sizes, changed inputs/draws and both embedding modes; output ownership also checked. Actual exit0.
- One fresh service executed C1 and C2, 128 timed requests per cohort and two warmup requests each. All **256 new timed requests** reached real stop/EOS. All **256 pairs of original/new PCM were directly byte-compared and equal**; original WAV format, PCM hash, bytes, duration, stream-byte coverage and codec-frame agreement were also checked.
- One server and two benchmark processes exited0; no forced kill. Source/model hashes stayed unchanged. The same one-card 910B3 environment and image digest are recorded in `identity-restored.json`.

The reference WAVs are retained original U002 candidate outputs, not a newly executed control. `c1-restored.json` and `c2-restored.json` contain new per-request records; `*-retained-reference.json` explicitly identifies the historical records. New benchmark runspecs bind the actual source snapshot. `audit-restored-local.json` is a local source/metadata consistency check, not a fresh NPU run or an independent read of private WAV files.

## Performance applicability

The original three-pair performance campaign remains bound to `890e64ecd0ede47dd395d280886688ccc3dfcfad`. It was not rerun or relabeled for this startup-validation correction. Fresh three-pair performance qualification for R1 remains **INCONCLUSIVE**; the new timings are diagnostic. This PASS applies to startup checks, actual predictor graphs and fixed-corpus C1/C2 output regression only. No new throughput-gain claim, independent speech-quality evaluation or production qualification is made.

## Prior local results and retained originals

12 focused CPU unittest cases passed. Local pytest had 40 passed, 14 skipped and 9 passing subtests; skipped accelerator tests were not PASS. Local full unittest discovery exited1 due unavailable model/accelerator imports (8 import errors, 6 skipped startup cases), correctly INCONCLUSIVE at that time. Its private-log hashes remain in `validation.json`.

Private originals remain outside Git in access-controlled storage: archive label `U002_R1_RESTORED_PRIVATE_ORIGINALS`, 124948480 bytes, SHA256 `e1c88ed5c05ba8c54599497d61b10acd69710368beef3fa566554ea5ca3fbb51`. They include the new WAVs, selected actual reference WAVs, raw logs, runspecs, operator evidence, source snapshot and executed runner/collector. Shared records omit connection identities, resolved paths, endpoints and input texts. No merge or production activation is part of this task.
