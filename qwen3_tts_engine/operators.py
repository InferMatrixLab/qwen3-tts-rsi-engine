"""Opt-in Ascend operators with explicit reference rounding semantics."""
import torch
import torch_npu


class AscendRMSNorm(torch.nn.Module):
    def __init__(self, reference, mode='reference-rounding', reference_single=False):
        super().__init__()
        self.reference = reference
        self.reference_single = reference_single
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon
        self.mode = mode
        self.register_buffer('unit_weight', torch.ones_like(self.weight, dtype=torch.float32))

    def forward(self, inputs):
        if self.reference_single and inputs.shape[0] == 1:
            return self.reference(inputs)
        if self.mode == 'native':
            return torch_npu.npu_rms_norm(inputs, self.weight, epsilon=self.variance_epsilon)[0]
        normalized = torch_npu.npu_rms_norm(inputs.float(), self.unit_weight,
                                           epsilon=self.variance_epsilon)[0]
        return normalized.to(inputs.dtype) * self.weight


def install_rms_norms(model, mode='reference-rounding', reference_single=False):
    replacements = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == 'Qwen3TTSRMSNorm':
                replacements.append((parent, name, child))
                setattr(parent, name, AscendRMSNorm(child, mode, reference_single))
    return replacements


def restore_modules(replacements):
    for parent, name, original in reversed(replacements):
        setattr(parent, name, original)
