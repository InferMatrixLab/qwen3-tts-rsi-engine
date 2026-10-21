"""Read-only one-request inputs; consumers copy into their own graph storage."""

from typing import NamedTuple


class SingleRequestInputs(NamedTuple):
    tokens: object
    hidden: object
    draws: object
    text: object
    uniforms: object
    seen: object
    positions: object
    rotary: object
    steps: object


def single_request_inputs(state):
    """Borrow tensor views and upload the three integer inputs together.

    Use only when the actual and graph batches are both one. Sources must stay
    alive and read-only until owner-stream consumers have copied these inputs.
    """
    import torch

    frame = state.frames
    if type(frame) is not int or not 0 <= frame < state.uniforms.shape[0] - 1:
        raise ValueError("Expected a frame with current and next uniform rows")
    integers = torch.tensor(
        [state.position, state.position + state.rope_delta, frame + 1],
        device=state.token.device,
        dtype=torch.long,
    )
    text = (
        state.trailing[:, frame : frame + 1]
        if frame < state.trailing.shape[1]
        else state.pad
    )
    return SingleRequestInputs(
        state.token,
        state.hidden,
        state.uniforms[frame : frame + 1],
        text,
        state.uniforms[frame + 1 : frame + 2, 0],
        state.seen,
        integers[:1],
        integers[1:2].view(1, 1, 1).expand(3, -1, -1),
        integers[2:3],
    )
