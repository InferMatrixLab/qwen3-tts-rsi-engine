"""Bounded eight-frame code history for independently advancing codec slots."""
import torch


class CodecCodeBuffer:
    def __init__(self, slots, codebooks, device):
        self.slots = slots
        self.codebooks = codebooks
        self.data = torch.zeros(slots, 8, codebooks, dtype=torch.long, device=device)
        self.rows = torch.arange(slots, device=device) * 8

    def write(self, codes, frame_indices):
        count = codes.shape[0]
        if codes.shape != (count, self.codebooks) or frame_indices.shape != (count,) or count > self.slots:
            raise ValueError('Code rows and frame indices must fit the slot arena')
        indices = self.rows[:count] + frame_indices.remainder(8)
        self.data.view(-1, self.codebooks).index_copy_(0, indices, codes)

    def read(self, slots, starts, length):
        if (not slots or len(slots) != len(starts) or not 1 <= length <= 8
                or min(slots) < 0 or max(slots) >= self.slots or min(starts) < 0):
            raise ValueError('Expected valid slots and a retained 1..8-frame interval')
        indices = torch.tensor([slot * 8 + (start + offset) % 8
                                for slot, start in zip(slots, starts) for offset in range(length)],
                               device=self.data.device)
        return self.data.view(-1, self.codebooks).index_select(0, indices).view(
            len(slots), length, self.codebooks).transpose(1, 2)

    def move(self, source, destination):
        if source != destination:
            self.data[destination].copy_(self.data[source])
