# Experimental dual-stream incremental Codec

`--codec-streams 2` lets independent incremental Codec groups overlap on two
Ascend NPU streams. The default is `--codec-streams 1`. Two streams require
`--continuous --incremental-codec` and the Triton Ascend runtime.

The grouping policy, group membership, batch sizes, chunk lengths, decoder
arithmetic and callback order stay the same. This option does not increase
`--codec-batch-size`. Qualification targets BF16 CustomVoice with Codec batch
size 4 and a maximum chunk of 8 frames; other configurations need their own
accelerator evidence.

Each lane has independent graph inputs, scratch storage and owned output
clones. Groups within an emit operation must use disjoint slots. A pure-copy
kernel updates only the selected state rows, including strided KV tails.
Contiguous source segments use linear vector loads and stores. The owner
stream prepares inputs and both lanes wait for it. An event after each owned
waveform clone permits delivery in original group order as soon as that group
completes. The owner fences both lanes before subsequent slot moves and resets,
including when a delivery callback raises. Callbacks must not re-enter model
work. The aggregate batched-transfer option retains its existing behavior.
The existing single-owner model thread remains responsible for all model work.
For batched admissions, the dual-stream path clears the contiguous newly
admitted Codec slots once per state buffer. Existing live slots remain intact;
single admissions and the single-stream path retain their per-request reset.

An emit with exactly one group replays its lane-0 pure-copy graph on the owner
stream and uses direct delivery without lane events. Each graph binds its write
function at construction; serving never mutates arena dispatch. Owner and lane
replays use the same graph inputs in stream order. Prewarming captures both
lanes: batches 1–4 and lengths 1–8 require 64 graphs instead of the default 32.
This increases graph storage and capture work. Ordinary native slot clearing is
retained; no additional reset graphs are captured.
Cold Triton compilation and steady-state service performance are different
measurements; neither should be inferred from the other.

## Verification

The component probe accepts a separately staged, unchanged baseline file:

```sh
python -m tools.probe_codec_dual_state \
  --model '<MODEL_PATH>' \
  --control-codec '<BASELINE_SOURCE>/qwen3_tts_engine/codec_slots.py' \
  --output '<PRIVATE_OUTPUT>/operator.json'
```

It checks selected-row copies, changed inputs, mixed and ragged groups, lane
reuse, waveform values, every selected and unselected state buffer, slot
move/reset and live-state prewarm. It checks output counts, progressive callback
order and copied values, and owner-stream snapshots after a callback exception.
Its component timings do not qualify service throughput or latency.
Singleton checks cover all 32 batch/length shapes, two changed-input replays
each, actual copied callback values and exception-state snapshots.
Native resets cover all 16 component-arena slots with changed state and check
every selected and unrelated buffer. Lane/owner/lane transitions are checked
without intermediate explicit synchronization or a new index upload before
owner replay. Service validation additionally exercises
the 64-slot arena and requires normal shutdown.

## Request admission

`--packed-kv-load` requires `--continuous --graph-prefill` without
`--reference-single`. It packs the existing prefill cache after all prefill
arithmetic and stores the target arena in one allocation with contiguous
per-layer views. Each request's true prefix is loaded by one native strided
copy. Padding, other slots, capacity-growth state and normal slot moves retain
their values. Packed source cache storage adds memory and a GPU copy; target
consolidation does not duplicate the arena.

`--graph-sampling` also covers the first semantic token using the existing
batch-1 graph, immutable zero seen/step inputs and owned outputs. The eager path
is retained when graph sampling is disabled. Seeded random-table generation and
the selected first uniform value are unchanged. The qualified service option
set must be read from source-bound evidence; component timings alone do not
qualify either admission or Codec changes.

RFC U005 requires original WAV comparisons, real EOS and complete streaming
bytes, clean process exits, source/model/environment binding, shared-device
telemetry, an A/A calibration and repeated service comparisons before a PR.
Missing evidence stays INCONCLUSIVE. The known 32-request regression corpus is
reused coverage, not unseen quality evaluation. The metadata auditor can be
imported without torch or accelerator runtimes and does not replace independent
reads of the original private WAV files.

`--single-request-views` requires `--continuous` and defaults off. When the
actual request count and selected graph batch are both one, it borrows read-only
token, hidden, text, uniform and seen views and uploads position, rotary position
and semantic step in one integer tensor. Each graph still copies its inputs
into private static storage and returns its original owned clones. Callers keep
the borrowed sources alive and read-only until those owner-stream copies have
consumed them. Multiple requests and padded graph batches retain their existing
preparation path. This option changes neither random tables nor floating-point
arithmetic.

The actual-tick probe compares a separately staged frozen parent, the new
default path and the enabled path, using independent state arenas with shared
read-only model weights. It checks original inputs, all request/KV/Codec state,
normal EOS, streamed PCM and a 1→2→1 transition with cancellation. Its two-slot
component arena does not replace full service validation at capacity 64.

```sh
python -m tools.probe_single_request_views \
  --model '<MODEL_PATH>' \
  --control-continuous '<FROZEN_PARENT>/qwen3_tts_engine/continuous.py' \
  --workload '<MAIN_128_CORPUS>' \
  --output '<PRIVATE_OUTPUT>/views-operator.json'
```

Design and search history: [RFC U005](https://github.com/InferMatrixLab/qwen3-tts-rsi-npu-engine/issues/6).
