from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.brain.simulation import NeuralSimulation
from src.connectome.graph import (
    attach_cell_types,
    build_graph,
)
from src.connectome.loader import (
    load_cell_types,
    load_connectome,
)


CONNECTIONS_PATH = (
    "data/raw/connections_princeton.csv.gz"
)

CELL_TYPES_PATH = (
    "data/raw/consolidated_cell_types.csv.gz"
)

VISUAL_TYPES = {
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
}

MAX_HOPS = 5
TOP_NEIGHBORS = 8
TOP_START_NEURONS = 30
MIN_SYNAPSES = 5

DN_PREFIXES = (
    "DN",
)


def is_visual(
    graph: nx.DiGraph,
    neuron_id: int,
) -> bool:
    cell_type = graph.nodes[
        neuron_id
    ].get("cell_type", "Unknown")

    return cell_type in VISUAL_TYPES


def is_descending(
    graph: nx.DiGraph,
    neuron_id: int,
) -> bool:
    cell_type = graph.nodes[
        neuron_id
    ].get("cell_type", "Unknown")

    return cell_type.startswith(
        DN_PREFIXES
    )


def get_neurons_by_type(
    graph: nx.DiGraph,
    cell_type: str,
) -> list[int]:
    return [
        neuron_id
        for neuron_id, data
        in graph.nodes(data=True)
        if data.get("cell_type")
        == cell_type
    ]


def get_strong_outgoing(
    graph: nx.DiGraph,
    neuron_id: int,
) -> list[tuple[int, int]]:
    """Return strongest outgoing actual neuron connections."""

    results = []

    for target in graph.successors(
        neuron_id
    ):
        data = graph[
            neuron_id
        ][target]

        syn_count = int(
            data.get("syn_count", 0)
        )

        if syn_count < MIN_SYNAPSES:
            continue

        results.append(
            (
                target,
                syn_count,
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results[:TOP_NEIGHBORS]


def build_visual_inputs(
    graph: nx.DiGraph,
) -> list[int]:
    """Select strong representative neurons from every T4/T5 type."""

    selected = []

    print()
    print(
        "=== Visual input neurons ==="
    )

    for cell_type in sorted(
        VISUAL_TYPES
    ):
        neurons = get_neurons_by_type(
            graph,
            cell_type,
        )

        candidates = []

        for neuron_id in neurons:
            outgoing = (
                get_strong_outgoing(
                    graph,
                    neuron_id,
                )
            )

            if outgoing:
                candidates.append(
                    (
                        neuron_id,
                        outgoing[0][1],
                    )
                )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        selected_type = [
            neuron_id
            for neuron_id, _
            in candidates[
                :TOP_START_NEURONS
            ]
        ]

        selected.extend(
            selected_type
        )

        print(
            f"{cell_type:<5} "
            f"total={len(neurons):,} "
            f"selected={len(selected_type)}"
        )

    return selected


def expand_circuit(
    graph: nx.DiGraph,
    input_neurons: list[int],
) -> nx.DiGraph:
    """Expand a real neuron-level circuit from visual inputs."""

    circuit = nx.DiGraph()

    frontier = set(
        input_neurons
    )

    visited = set()

    for neuron_id in input_neurons:
        circuit.add_node(
            neuron_id,
            cell_type=graph.nodes[
                neuron_id
            ].get(
                "cell_type",
                "Unknown",
            ),
        )

    for hop in range(
        MAX_HOPS
    ):
        print()
        print(
            f"Expanding hop {hop + 1}..."
        )

        next_frontier = set()

        for neuron_id in frontier:
            if neuron_id in visited:
                continue

            visited.add(
                neuron_id
            )

            for target, syn_count in (
                get_strong_outgoing(
                    graph,
                    neuron_id,
                )
            ):
                target_type = graph.nodes[
                    target
                ].get(
                    "cell_type",
                    "Unknown",
                )

                circuit.add_node(
                    target,
                    cell_type=target_type,
                )

                circuit.add_edge(
                    neuron_id,
                    target,
                    syn_count=syn_count,
                )

                next_frontier.add(
                    target
                )

        frontier = next_frontier

        print(
            f"  nodes: "
            f"{circuit.number_of_nodes():,}"
        )

        print(
            f"  edges: "
            f"{circuit.number_of_edges():,}"
        )

        if not frontier:
            break

    return circuit


def summarize_outputs(
    circuit: nx.DiGraph,
) -> None:
    """Summarize descending neurons reached by the circuit."""

    outputs = defaultdict(
        lambda: {
            "neurons": 0,
            "incoming_synapses": 0,
        }
    )

    for neuron_id, data in circuit.nodes(
        data=True
    ):
        cell_type = data.get(
            "cell_type",
            "Unknown",
        )

        if not cell_type.startswith(
            DN_PREFIXES
        ):
            continue

        incoming = circuit.in_edges(
            neuron_id,
            data=True,
        )

        total_synapses = sum(
            int(
                edge_data.get(
                    "syn_count",
                    0,
                )
            )
            for _, _, edge_data
            in incoming
        )

        outputs[cell_type][
            "neurons"
        ] += 1

        outputs[cell_type][
            "incoming_synapses"
        ] += total_synapses

    rows = []

    for cell_type, data in outputs.items():
        rows.append(
            (
                cell_type,
                data["neurons"],
                data["incoming_synapses"],
            )
        )

    rows.sort(
        key=lambda row: (
            row[2],
            row[1],
        ),
        reverse=True,
    )

    print()
    print(
        "=== Descending outputs ==="
    )
    print()

    if not rows:
        print(
            "No descending neurons reached."
        )
        return

    for index, (
        cell_type,
        neurons,
        synapses,
    ) in enumerate(
        rows[:30],
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{cell_type:<10} "
            f"neurons={neurons:<4} "
            f"incoming="
            f"{synapses:,}"
        )


def simulate(
    circuit: nx.DiGraph,
    input_neurons: list[int],
) -> None:
    """Run the expanded real-neuron circuit."""

    print()
    print(
        "=== Circuit simulation ==="
    )

    simulation = NeuralSimulation(
        circuit,
        threshold=1.0,
        decay=0.9,
        synapse_scale=10.0,
    )

    simulation.stimulate_many(
        input_neurons,
        signal=1.0,
    )

    history = simulation.run(
        steps=12
    )

    print()

    for step, fired in enumerate(
        history
    ):
        visual_count = sum(
            1
            for neuron_id in fired
            if is_visual(
                circuit,
                neuron_id,
            )
        )

        dn_count = sum(
            1
            for neuron_id in fired
            if is_descending(
                circuit,
                neuron_id,
            )
        )

        print(
            f"Step {step:2}: "
            f"fired={len(fired):,} "
            f"visual={visual_count:,} "
            f"DN={dn_count:,}"
        )

    dn_fired = defaultdict(int)

    for fired in history:
        for neuron_id in fired:
            if is_descending(
                circuit,
                neuron_id,
            ):
                cell_type = circuit.nodes[
                    neuron_id
                ].get(
                    "cell_type",
                    "Unknown",
                )

                dn_fired[
                    cell_type
                ] += 1

    print()
    print(
        "=== Fired descending neurons ==="
    )

    if not dn_fired:
        print(
            "No DN neurons fired."
        )
        return

    ranked = sorted(
        dn_fired.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for index, (
        cell_type,
        count,
    ) in enumerate(
        ranked[:30],
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{cell_type:<10} "
            f"firings={count}"
        )


def main() -> None:
    print(
        "Loading connectome..."
    )

    connections = load_connectome(
        CONNECTIONS_PATH
    )

    cell_types = load_cell_types(
        CELL_TYPES_PATH
    )

    graph = build_graph(
        connections
    )

    attach_cell_types(
        graph,
        cell_types,
    )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges():,}"
    )

    input_neurons = build_visual_inputs(
        graph
    )

    print()
    print(
        f"Total visual inputs: "
        f"{len(input_neurons):,}"
    )

    print()
    print(
        "Building real neuron-level circuit..."
    )

    circuit = expand_circuit(
        graph,
        input_neurons,
    )

    print()
    print(
        "=== Circuit summary ==="
    )

    print(
        f"Nodes: "
        f"{circuit.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{circuit.number_of_edges():,}"
    )

    summarize_outputs(
        circuit
    )

    simulate(
        circuit,
        input_neurons,
    )


if __name__ == "__main__":
    main()