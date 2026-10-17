"""Bounded graph batch selection for a changing active request count."""


def decode_batch_size(active, max_slots, step=0):
    if not 1 <= active <= max_slots or max_slots & (max_slots - 1):
        raise ValueError('Active count must fit a positive power-of-two slot arena')
    if step < 0 or (step and (step & (step - 1) or step > max_slots)):
        raise ValueError('Decode bucket step must be zero or a power of two within the arena')
    if not step or active <= step:
        return 1 << (active - 1).bit_length()
    return ((active + step - 1) // step) * step
