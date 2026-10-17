"""Experimental packed gate/up projection with separately selectable SwiGLU fusion."""
import torch


class PackedMLP(torch.nn.Module):
    def __init__(self, reference, fused=False):
        super().__init__()
        if reference.config.hidden_act != 'silu':
            raise ValueError('Packed MLP requires the SiLU activation')
        if reference.gate_proj.bias is not None or reference.up_proj.bias is not None:
            raise ValueError('Packed MLP expects bias-free projections')
        self.weight = torch.nn.Parameter(torch.cat((reference.gate_proj.weight.detach(),
                                                   reference.up_proj.weight.detach()), dim=0), requires_grad=False)
        self.down_proj = reference.down_proj
        self.act_fn = reference.act_fn
        self.fused = fused

    def forward(self, hidden):
        projected = torch.nn.functional.linear(hidden, self.weight)
        if self.fused:
            import torch_npu
            hidden = torch_npu.npu_swiglu(projected)
        else:
            gate, up = projected.chunk(2, dim=-1)
            hidden = self.act_fn(gate) * up
        return self.down_proj(hidden)


def install_packed_mlp(model, fused=False):
    replacements = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child).__name__ == 'Qwen3TTSTalkerTextMLP':
                replacements.append((parent, name, child))
                setattr(parent, name, PackedMLP(child, fused=fused))
    return replacements
