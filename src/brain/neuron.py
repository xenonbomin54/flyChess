from __future__ import annotations


class Neuron:
    """Simple rate-based neuron for connectome simulation."""

    def __init__(
        self,
        neuron_id: int,
        threshold: float = 1.0,
        decay: float = 0.9,
    ):
        self.neuron_id = neuron_id
        self.threshold = threshold
        self.decay = decay
        self.activity = 0.0

    def receive(self, signal: float) -> None:
        """Accumulate incoming neural activity."""
        self.activity += signal

    def step(self) -> bool:
        """Advance one simulation step and return whether the neuron fired."""
        fired = self.activity >= self.threshold

        if fired:
            self.activity = 0.0
        else:
            self.activity *= self.decay

        return fired