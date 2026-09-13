from __future__ import annotations

import networkx as nx

from src.brain.neuron import Neuron


class NeuralSimulation:
    """Run activity propagation on a connectome graph."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.neurons = {
            neuron_id: Neuron(neuron_id)
            for neuron_id in graph.nodes
        }

    def stimulate(self, neuron_id: int, signal: float = 1.0) -> None:
        """Inject activity into a neuron."""
        if neuron_id not in self.neurons:
            raise ValueError(f"Unknown neuron: {neuron_id}")

        self.neurons[neuron_id].receive(signal)

    def _connection_signal(self, data: dict) -> float:
        """Calculate signal strength from a connectome connection."""
        syn_count = data.get("syn_count", 1)

        # Current prototype uses synapse count as signal strength.
        # Neurotransmitter effects are preserved in the graph data
        # but are not yet applied to the simulation.
        return min(syn_count / 10.0, 1.0)

    def step(self) -> list[int]:
        """Run one simulation step."""
        fired = []

        # 1. Determine which neurons fire.
        for neuron_id, neuron in self.neurons.items():
            if neuron.step():
                fired.append(neuron_id)

        # 2. Propagate activity through outgoing connections.
        for neuron_id in fired:
            for target in self.graph.successors(neuron_id):
                data = self.graph[neuron_id][target]
                signal = self._connection_signal(data)

                if signal != 0:
                    self.neurons[target].receive(signal)

        return fired