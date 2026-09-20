from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NeuralOutput:
    """Summarized output of the neural simulation."""

    strongest_neuron: int | None
    strongest_activity: int
    total_activity: int
    active_neurons: int

    def has_activity(self) -> bool:
        return self.total_activity > 0


class BrainOutput:
    """Convert neural firing history into a compact output signal."""

    def __init__(
        self,
        output_neurons: list[int] | None = None,
    ):
        self.output_neurons = set(
            output_neurons or []
        )

    def set_output_neurons(
        self,
        neuron_ids: list[int],
    ) -> None:
        """Set neurons that should be treated as output neurons."""
        self.output_neurons = set(neuron_ids)

    def analyze(
        self,
        history: list[list[int]],
    ) -> NeuralOutput:
        """Analyze firing history of the selected output neurons."""

        counts: dict[int, int] = defaultdict(int)

        for fired in history:
            for neuron_id in fired:
                if (
                    not self.output_neurons
                    or neuron_id in self.output_neurons
                ):
                    counts[neuron_id] += 1

        if not counts:
            return NeuralOutput(
                strongest_neuron=None,
                strongest_activity=0,
                total_activity=0,
                active_neurons=0,
            )

        strongest_neuron = max(
            counts,
            key=counts.get,
        )

        return NeuralOutput(
            strongest_neuron=strongest_neuron,
            strongest_activity=counts[
                strongest_neuron
            ],
            total_activity=sum(
                counts.values()
            ),
            active_neurons=len(counts),
        )

    def analyze_by_type(
        self,
        history: list[list[int]],
        neuron_to_type: dict[int, str],
    ) -> dict[str, int]:
        """Aggregate firing activity by neuron type."""

        counts: dict[str, int] = defaultdict(int)

        for fired in history:
            for neuron_id in fired:
                if (
                    self.output_neurons
                    and neuron_id not in self.output_neurons
                ):
                    continue

                neuron_type = neuron_to_type.get(
                    neuron_id,
                    "Unknown",
                )

                counts[neuron_type] += 1

        return dict(
            sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def get_action_signal(
        self,
        output: NeuralOutput,
        threshold: int = 1,
    ) -> bool:
        """Convert neural activity into a binary action signal."""

        return (
            output.total_activity >= threshold
        )