# Final acceptance: PASS

The frozen Qwen3-TTS CustomVoice contract is satisfied on one Ascend 910B1.
Three independent runs of each framework cover C1/2/4/8/16/32/64 with 128
requests per concurrency. All **8,064 requests** succeeded, all **nine servers
exited 0**, and all **63 cohorts** passed retained-audio, timing and scoring
integrity checks. All three candidate quality loops also exited 0.

Every candidate trial exceeds twice the highest selected individual baseline
throughput at the same concurrency. The smallest observed ratio is **2.047683×**
(C32). This is an observed-run result, not a statistical lower confidence bound.

All 2,688 candidate waveforms were independently scored. Every trial/concurrency
passes the frozen English WER, Chinese CER and speaker-similarity margins even
against the strongest observed quality references across all six baseline runs.
Exact WAV identity is recorded separately and was not used to replace scoring.

## Requirement evidence

| Requirement | Evidence |
|---|---|
| Independent CustomVoice engine | [Runtime source equivalence and import audit](results/engine-source-equivalence.json), [reproduction](../../REPRODUCE.md) |
| OmniRSI, Strata and Ascend experience | [Source and mechanism provenance](plan/sources.md), [retained experiment history](round1.md) |
| Pinned baseline revisions and workload | [All 63 run specifications and lifecycle audit](results/final-source-workload-lifecycle-audit.json), [run selection](plan/final-comparison-runs.json) |
| Identical pinned model files | [Post-performance model rehash](results/model-recheck-after-final-performance.json) |
| Four requested metrics at every concurrency | [Three-framework table](same-card-performance.md), [all raw trial values](results/final-three-arm-performance.json), [charts](same-card-three-trial-charts/README.md) |
| Three repeats, full quality, zero failures and 2× throughput | [Full numerical and cohort audit](results/final-numerical-audit.json) |
| Request-level records available for inspection | [Compressed evidence](final-request-records.tar.gz), [archive SHA256 and verification](final-request-records.tar.gz.json) |
| Regression checks | [Recorded 57-test summary](results/direct-prompt-test-summary.json); target-NPU end-to-end evidence is separate above |

The archive contains request timings and PCM hashes, all transcripts and speaker
scores, run specifications and available provenance. Its internal manifest hashes
every member, and the exporter re-read every member to verify the saved archive.
Weights, WAV files and verbose logs remain at the retained local paths; they are
not included in this compact public archive.

## Qualifications

SGLang is pinned main plus the recorded type-import compatibility patch, not
untouched main. Both baseline configurations use the recorded jemalloc workaround;
the upstream allocator issue is not claimed fixed. First-round vLLM used an older
runner without automatic environment/provenance capture; physical NPU 0 is recorded
in the contemporaneous campaign report. All later runs record it directly.

The host was shared and trials were sequential. Sample SD describes observed
variation; there is no statistical-significance or global-optimality claim.
Automatic ASR and speaker measures do not establish subjective naturalness.
The conclusion applies to this pinned model, workload, sampling and measured
baseline configurations. See the unchanged [contract](plan/contract.json),
[quality margins](plan/quality-gates.json) and [environment](plan/environment.md).
