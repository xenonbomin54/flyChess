from __future__ import annotations

import networkx as nx

from src.brain.neuron import Neuron


class NeuralSimulation:
    """Run activity propagation on a connectome graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        threshold: float = 1.0,
        decay: float = 0.9,
        synapse_scale: float = 10.0,
    ):
        self.graph = graph
        self.threshold = threshold
        self.decay = decay
        self.synapse_scale = synapse_scale

        self.neurons = {
            neuron_id: Neuron(
                neuron_id=neuron_id,
                threshold=threshold,
                decay=decay,
            )
            for neuron_id in graph.nodes
        }

        self.last_fired: list[int] = []

    def stimulate(
        self,
        neuron_id: int,
        signal: float = 1.0,
    ) -> None:
        """Inject activity into a neuron."""

        if neuron_id not in self.neurons:
            raise ValueError(
                f"Unknown neuron: {neuron_id}"
            )

        self.neurons[neuron_id].receive(signal)

    def stimulate_many(
        self,
        neuron_ids: list[int],
        signal: float = 1.0,
    ) -> None:
        """Stimulate multiple neurons at once."""

        for neuron_id in neuron_ids:
            self.stimulate(
                neuron_id,
                signal,
            )

    def _connection_signal(
        self,
        data: dict,
    ) -> float:
        """Convert synapse count into a bounded neural signal."""

        syn_count = float(
            data.get("syn_count", 1)
        )

        return min(
            syn_count / self.synapse_scale,
            1.0,
        )

    def step(self) -> list[int]:
        """Advance the network by one simulation step."""

        fired = []

        # First evaluate every neuron using its current activity.
        for neuron_id, neuron in self.neurons.items():
            if neuron.step():
                fired.append(neuron_id)

        # Then propagate fired activity to downstream neurons.
        for neuron_id in fired:
            for target in self.graph.successors(
                neuron_id
            ):
                data = self.graph[
                    neuron_id
                ][target]

                signal = self._connection_signal(
                    data
                )

                if signal > 0:
                    self.neurons[
                        target
                    ].receive(signal)

        self.last_fired = fired

        return fired

    def run(
        self,
        steps: int = 10,
    ) -> list[list[int]]:
        """Run multiple simulation steps."""

        history = []

        for _ in range(steps):
            fired = self.step()
            history.append(fired)

        return history

    def get_activity(
        self,
        neuron_id: int,
    ) -> float:
        """Get current activity of a neuron."""

        if neuron_id not in self.neurons:
            raise ValueError(
                f"Unknown neuron: {neuron_id}"
            )

        return self.neurons[
            neuron_id
        ].activity

    def get_firing_count(
        self,
        history: list[list[int]],
        neuron_ids: list[int],
    ) -> dict[int, int]:
        """Count how many times selected neurons fired."""

        counts = {
            neuron_id: 0
            for neuron_id in neuron_ids
        }

        for fired in history:
            for neuron_id in fired:
                if neuron_id in counts:
                    counts[neuron_id] += 1

        return counts