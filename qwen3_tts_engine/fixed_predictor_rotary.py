"""Capture-local coefficients for the predictor's immutable codebook positions."""
from contextlib import contextmanager

import torch


class FixedPredictorRotary(torch.nn.Module):
    """Read-only replacement, scoped to warmup/capture of one predictor graph.

    Positions are explicitly owned objects, not pointers into reusable storage.
    Never install this adapter for ordinary generation or dynamic talker positions.
    """

    @torch.inference_mode()
    def __init__(self, reference, hidden, positions):
        super().__init__()
        if reference.training or getattr(reference, "rope_type", None) != "default":
            raise ValueError("Fixed predictor RoPE requires evaluation with default RoPE")
        self.dtype = hidden.dtype
        self.device = hidden.device
        self.entries = {}
        for position in positions:
            coefficients = reference(hidden, position)
            self.entries[id(position)] = (position, coefficients)

    def forward(self, hidden, position_ids):
        entry = self.entries.get(id(position_ids))
        if entry is None or entry[0] is not position_ids:
            raise ValueError("Fixed predictor RoPE received an unowned position tensor")
        if hidden.dtype != self.dtype or hidden.device != self.device:
            raise ValueError("Fixed predictor RoPE dtype/device changed")
        return entry[1]

    @contextmanager
    def installed(self, model):
        original = model.rotary_emb
        model.rotary_emb = self
        try:
            yield
        finally:
            model.rotary_emb = original
