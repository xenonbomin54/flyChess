from __future__ import annotations

import networkx as nx

from src.brain.neuron import Neuron


class NeuralSimulation:
    """Run activity propagation on a connectome graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        threshold: float = 0.1,
        decay: float = 0.9,
    ):
        self.graph = graph
        self.threshold = threshold
        self.decay = decay

        self.neurons = {
            neuron_id: Neuron(
                neuron_id=neuron_id,
                threshold=threshold,
                decay=decay,
            )
            for neuron_id in graph.nodes
        }

        self.outgoing_totals = {}

        for neuron_id in graph.nodes:
            total = 0

            for target in graph.successors(
                neuron_id
            ):
                data = graph[
                    neuron_id
                ][target]

                total += int(
                    data.get(
                        "syn_count",
                        0,
                    )
                )

            self.outgoing_totals[
                neuron_id
            ] = total

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

        self.neurons[
            neuron_id
        ].receive(signal)

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
        source: int,
        target: int,
    ) -> float:
        """Calculate normalized synaptic transmission."""

        data = self.graph[
            source
        ][target]

        syn_count = float(
            data.get(
                "syn_count",
                0,
            )
        )

        total_output = float(
            self.outgoing_totals.get(
                source,
                0,
            )
        )

        if total_output <= 0:
            return 0.0

        return syn_count / total_output

    def step(self) -> list[int]:
        """Advance one simulation step."""

        fired = []

        # Evaluate all neurons using their current activity.
        for neuron_id, neuron in self.neurons.items():
            if neuron.step():
                fired.append(
                    neuron_id
                )

        # Propagate activity after the firing decision.
        for neuron_id in fired:
            for target in self.graph.successors(
                neuron_id
            ):
                signal = self._connection_signal(
                    neuron_id,
                    target,
                )

                if signal <= 0:
                    continue

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
            history.append(
                self.step()
            )

        return history

    def get_activity(
        self,
        neuron_id: int,
    ) -> float:
        """Get current activity."""

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
        """Count firing events for selected neurons."""

        counts = {
            neuron_id: 0
            for neuron_id in neuron_ids
        }

        for fired in history:
            for neuron_id in fired:
                if neuron_id in counts:
                    counts[
                        neuron_id
                    ] += 1

        return counts