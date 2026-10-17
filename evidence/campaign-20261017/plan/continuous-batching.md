# Continuous batching implementation plan

The whole-batch Hugging Face generation adapter cannot meet the requested
per-concurrency target: the experimental native-fusion sweep measured
25.829 frames/s at C1, 36.574 at C2, 45.254 at C4, and 63.510 at C8.
Completed rows continue consuming the batch until its longest request finishes.
A C1 result alone is not acceptance.

Next implementation keeps the existing bounded HTTP/audio transport and replaces
the model's whole-request generation loop with an iteration scheduler:

1. Reuse the official CustomVoice prompt construction by intercepting its
   `talker.generate` invocation to capture prepared embeddings, masks, trailing
   text embeddings and generation settings. Do not run its generation loop.
2. Run eager prefill for newly admitted requests, copying each request's KV,
   final hidden state and initial semantic logits into a packed active-slot arena.
3. Capture single-step talker compute for active batch buckets. KV writes use
   per-row position tensors, with masks and rotary positions supplied per row.
   Keep all capture constants and arena storage alive. Grow sequence capacity
   explicitly rather than silently truncating output.
4. On EOS, flush that request's codec tail and complete its HTTP future
   immediately. Compact active slots by moving the last active row into a hole;
   admit queued requests on the next iteration. Preserve other requests' KV,
   sampling history, text position and output accounting during compaction.
5. Use per-request seeded uniform tables indexed by frame and codebook to make
   sampling independent of slot changes. Inverse-CDF sampling after the same
   temperature, top-k threshold, suppression and repetition processing preserves
   the categorical distribution; it does not reproduce torch.multinomial's
   seed-to-token mapping. Normalize CDF totals and use strict upper bounds so
   zero-probability tokens cannot be selected when a uniform draw equals zero.
6. The predictor runs fixed-depth graph compute with per-row uniform inputs.
   An async service tick submits work to the sole NPU owner thread, so arrivals
   can be admitted between decode iterations without moving model ownership.
7. Batch codec windows in small groups with bounded graph shapes. Align context
   with the decoder's 72-frame sliding window. Existing 25-frame streaming
   results stay identified by source commit and must not be relabeled.

Validation must cover slot reuse, heterogeneous prompt/output lengths, early EOS,
request cancellation, capacity growth, deterministic per-request sampling under
slot permutations, no missing/duplicated audio, clean shutdown and full ASR/SIM.
Measure the same complete C1-C64 corpus again. The original 2x target and all
quality/lifecycle requirements remain unchanged.
