# U002 review R2: CPU startup coverage

Status: **PASS for CPU startup coverage and fresh A2 regressions**. This addresses the additional review request to run startup checks in ordinary CPU CI, rather than skipping them when optional accelerator/model packages are unavailable.

The tested uncommitted source snapshot is `255ecce8d849165e7a03879eca1e661ad9dc013dd35241f71bb370e2ea6fff77`, based on `cde3a52d81e9811d887941f23416855dbd5150d6`. All 90 measured Python blobs match the working tree; the publication check also verifies the final committed blobs. Exactly two Python files differ from that parent: `continuous.py` and `test_predictor_startup.py`.

## Focused changes

Move the accelerator-dependent `CustomVoiceModel` import from module scope into `ContinuousModel.__init__`. `UniformPredictor` and its metadata-only constructor guards can now be imported and exercised without torch_npu/qwen_tts. Full model construction still loads its normal accelerator dependencies. No attention, sampling, rotary arithmetic or graph replay operation changes.

Remove the class-wide skip from all six startup tests. For no-capture assertions, supply a temporary forbidden-Stream stub on `torch.npu`, including when that attribute is absent from CPU torch. The stub raises if capture is attempted; it does not simulate successful NPU execution. Actual predictor graphs and service outputs are separately validated on A2.

## Actual CPU results

The local environment has neither `torch_npu` nor `qwen_tts`. All **6 startup cases passed without skips**, including structure/training/RoPE rejection, default-off compatibility, capture-time revalidation and service-startup failure before ready/worker creation. CPU testing still requires torch, transformers and normal project dependencies (numpy/aiohttp).

```bash
python -m unittest discover -s tests -p test_predictor_startup.py -v
```

Local pytest: **46 passed, 8 skipped**, 17 passing subtests. The remaining skips are other existing accelerator-dependent modules, not the startup class. Full local unittest discovery exited1 with 8 dependency-import errors; its result remains INCONCLUSIVE. Raw-log hashes, per-case results and optional-package absence are in `local-cpu.json`. Skips are not PASS.

## Fresh A2 validation of this exact snapshot

- Required unittest discovery: **18 passed**; full pytest: **75 passed**, 14 existing warnings, 17 passing subtests. The same six startup cases ran and passed on A2 (`startup-cases-a2.json`).
- Real-model predictor probe: **40 exact graph comparisons**, across five batch sizes, changed inputs/draws and both embedding modes; returned-output ownership also checked.
- A fresh service generated C1/C2 with 128 timed requests per cohort, two warmups each. All **256 new timed requests** reached real EOS; **256 original/new PCM pairs were directly byte-compared and equal**. WAV format, PCM SHA256/length/duration, stream bytes and codec-frame counts were checked against retained original U002 candidate WAVs.
- Server and both benchmark processes exited0, no forced kill. Source/model hashes were unchanged throughout, and the used NPU was idle afterward. Environment/model/source details are in `identity-a2.json`.

The retained reference is not a freshly executed control. `c1-a2.json`/`c2-a2.json` are new actual records; `*-retained-reference.json` explicitly identifies the old records. `audit-local.json` checks published metadata/source consistency only, not new inference or independent access to private audio.

## Performance and historical applicability

No new three-pair performance qualification was run: that item remains INCONCLUSIVE for this revision. Original throughput measurements remain bound to `890e64ecd0ede47dd395d280886688ccc3dfcfad`. R1 startup-validation records remain bound to their tested implementation and are not reused as this snapshot's A2 results. This revision's PASS is scoped to CPU coverage, startup behavior, actual graph comparisons and C1/C2 fixed-corpus output regression; it does not add throughput, independent speech-quality or production claims.

Private raw originals: archive label `U002_R2_CPU_STARTUP_PRIVATE_ORIGINALS`, 124948480 bytes, SHA256 `c4177aab4b12f1fcbb52bf7e2835fdb11078996b2910e567ab27f99161deb775`. New/reference WAVs, logs, runspecs, actual runner/collector and source snapshot remain access-controlled outside Git. Public records omit connection identities, resolved paths, endpoints, raw exception text and input texts. No merge or production activation is included.
