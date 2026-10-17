# Incremental codec investigation

The continuous C32 trace has 41 codec calls in 16 ticks. Inclusive CPU ranges
attribute 5.15 seconds to codec calls out of 8.18 seconds of tick ranges.
These ranges include asynchronous device waits and profiler overhead; they are
not additive component latency or a service benchmark. Convolution, transpose
convolution and pointwise audio operations are prominent in device kernel time.

The current window decoder repeatedly processes up to 72 old codec frames for
each 8 new frames. The pretrained decoder is causal: stride-one convolutions,
causal transpose convolutions, and eight transformer layers with a 72-frame
sliding attention window. Reusing their state can eliminate repeated history
computation without dropping codebooks or shortening generated speech.

`incremental_codec.py` is an eager correctness implementation, not enabled in
the service. Each stream owns convolution input history, transpose-convolution
overlap, transformer KV state and an absolute frame position. Overlap tails
exclude bias before they are added to the next chunk. It reuses original weights
and preserves the pretrained modules without monkey-patching their methods.

Validation retained separately:

- CPU tests compare complete versus arbitrarily split dilated, pointwise and
  transpose convolutions, including interleaved independent streams and reset.
- Synthetic 101-frame sequences cross the attention-window boundary at batch
  sizes one and two. BF16, default FP32 and strict FP32 are separate diagnostics.
- Four real English/Chinese prompts retain original and incremental WAVs,
  codec tokens, checksums and sample counts. Output is numerically approximate;
  full-corpus ASR/speaker gates still apply.

The next implementation must batch new chunks and capture NPU graphs. A shared
slot arena should retain convolution tails, the last 71 KV positions per codec
layer, and absolute frame counters. Each graph selects request slots with tensor
indices, processes only new frames, then writes updated state back. Attention
masks must exclude uninitialized history and enforce the 72-frame window for
every query, including chunks that cross the window boundary. Keep graph-owned
buffers alive and restore arena state after capture warmups. Slot compaction,
cancellation and reuse must move or clear codec state along with talker state.

Validate interleaved requests and reset before wiring this into the continuous
service. Measure matched runs after independent audio scoring; the eager probe
does not establish any speedup or acceptance.
