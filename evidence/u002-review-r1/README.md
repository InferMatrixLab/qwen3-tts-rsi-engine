# U002 review R1 validation

Fresh A2 validation: **INCONCLUSIVE**. Code synchronization did not complete: the configured SSH connection timed out (exit255). No A2 test or service run has been executed for this review revision. Do not merge on the basis of this record. The original three-pair performance campaign remains bound to `890e64ecd0ede47dd395d280886688ccc3dfcfad`; its data is retained unchanged and is not relabeled as new-head validation.

## Review corrections

- Opt-in `UniformPredictor` construction resolves `reference.model.rotary_emb` and rejects missing/wrapped structures, training and non-default RoPE before service startup completes. The shared validator inspects metadata only. Capture validates again; disabled mode does not inspect the wrapper.
- Recommend enabling only at C1–C2. At C≥4 expect no useful throughput benefit: all original higher-load rows had two negative repeats and conservative p95 TTFA regressed up to 4.30% at C16 and 3.96% at C32. Default remains disabled.
- New tests cover constructor success without coefficient computation/capture, incompatible structures/configuration, default-off compatibility, capture-time revalidation and service startup failure before ready/worker creation.

## Actual local checks

12 focused CPU unittest cases passed. Local pytest completed with 40 passed, 14 skipped and 9 passing subtests. Skipped accelerator-dependent tests are not PASS. Full unittest discovery exited1 due unavailable model/accelerator imports (8 import errors, 6 skipped startup cases); it is INCONCLUSIVE, not PASS. Redacted summaries and private-log SHA256 values are in `validation.json`.

## Pending A2 checks

After connection restoration: synchronize this revision, run `python -m unittest discover -s tests -v`, the full pytest suite and the existing real-model predictor graph probe; then verify actual C1/C2 service PCM against retained original WAVs. Preserve process exits, actual source snapshots and original audio outside Git. Publish a separate validation record without erasing this failed connection attempt. This scope does not repeat or claim a fresh three-pair performance qualification.
