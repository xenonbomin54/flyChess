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

VISUAL_GROUPS = {
    "T4a": {"T4a"},
    "T4b": {"T4b"},
    "T4c": {"T4c"},
    "T4d": {"T4d"},
    "T5a": {"T5a"},
    "T5b": {"T5b"},
    "T5c": {"T5c"},
    "T5d": {"T5d"},
    "T4": {
        "T4a",
        "T4b",
        "T4c",
        "T4d",
    },
    "T5": {
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    },
    "T4+T5": {
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    },
}

TOP_INPUTS_PER_TYPE = 30
TOP_NEIGHBORS = 8
MAX_HOPS = 5
MIN_SYNAPSES = 5

SIMULATION_STEPS = 12

THRESHOLD = 0.1
DECAY = 0.9


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
    connections = []

    for target in graph.successors(
        neuron_id
    ):
        data = graph[
            neuron_id
        ][target]

        synapses = int(
            data.get(
                "syn_count",
                0,
            )
        )

        if synapses < MIN_SYNAPSES:
            continue

        connections.append(
            (
                target,
                synapses,
            )
        )

    connections.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return connections[
        :TOP_NEIGHBORS
    ]


def select_input_neurons(
    graph: nx.DiGraph,
) -> dict[str, list[int]]:
    selected = {}

    all_types = set()

    for group in VISUAL_GROUPS.values():
        all_types.update(group)

    for cell_type in sorted(
        all_types
    ):
        neurons = get_neurons_by_type(
            graph,
            cell_type,
        )

        candidates = []

        for neuron_id in neurons:
            outgoing = get_strong_outgoing(
                graph,
                neuron_id,
            )

            if not outgoing:
                continue

            total = sum(
                synapses
                for _, synapses
                in outgoing
            )

            candidates.append(
                (
                    neuron_id,
                    total,
                )
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        selected[
            cell_type
        ] = [
            neuron_id
            for neuron_id, _
            in candidates[
                :TOP_INPUTS_PER_TYPE
            ]
        ]

    return selected


def build_circuit(
    graph: nx.DiGraph,
    input_neurons: list[int],
) -> nx.DiGraph:
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

    for _ in range(
        MAX_HOPS
    ):
        next_frontier = set()

        for neuron_id in frontier:
            if neuron_id in visited:
                continue

            visited.add(
                neuron_id
            )

            for target, synapses in (
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
                    syn_count=synapses,
                )

                next_frontier.add(
                    target
                )

        frontier = next_frontier

        if not frontier:
            break

    return circuit


def get_dn_type(
    graph: nx.DiGraph,
    neuron_id: int,
) -> str | None:
    cell_type = graph.nodes[
        neuron_id
    ].get(
        "cell_type",
        "Unknown",
    )

    if cell_type.startswith(
        "DN"
    ):
        return cell_type

    return None


def run_experiment(
    circuit: nx.DiGraph,
    input_neurons: list[int],
) -> dict:
    simulation = NeuralSimulation(
        circuit,
        threshold=THRESHOLD,
        decay=DECAY,
    )

    simulation.stimulate_many(
        input_neurons,
        signal=1.0,
    )

    history = simulation.run(
        steps=SIMULATION_STEPS
    )

    total_fired = sum(
        len(fired)
        for fired in history
    )

    dn_fired = defaultdict(int)
    dn_first_step = {}

    for step, fired in enumerate(
        history
    ):
        for neuron_id in fired:
            cell_type = get_dn_type(
                circuit,
                neuron_id,
            )

            if cell_type is None:
                continue

            dn_fired[
                cell_type
            ] += 1

            if neuron_id not in dn_first_step:
                dn_first_step[
                    neuron_id
                ] = step

    return {
        "total_fired": total_fired,
        "unique_dn_neurons": len(
            dn_first_step
        ),
        "dn_total": sum(
            dn_fired.values()
        ),
        "first_dn_step": (
            min(dn_first_step.values())
            if dn_first_step
            else None
        ),
        "dn_fired": dict(
            dn_fired
        ),
        "history": history,
    }


def print_top_outputs(
    result: dict,
) -> None:
    ranked = sorted(
        result["dn_fired"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for index, (
        cell_type,
        count,
    ) in enumerate(
        ranked[:10],
        start=1,
    ):
        print(
            f"    {index:2}. "
            f"{cell_type:<10} "
            f"{count:>4}"
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

    print()
    print(
        "Selecting visual input neurons..."
    )

    selected = select_input_neurons(
        graph
    )

    for cell_type in sorted(
        selected
    ):
        print(
            f"  {cell_type:<4} "
            f"{len(selected[cell_type])}"
        )

    all_inputs = []

    for neurons in selected.values():
        all_inputs.extend(
            neurons
        )

    all_inputs = list(
        dict.fromkeys(
            all_inputs
        )
    )

    print()
    print(
        "Building shared circuit..."
    )

    circuit = build_circuit(
        graph,
        all_inputs,
    )

    print(
        f"Circuit nodes: "
        f"{circuit.number_of_nodes():,}"
    )

    print(
        f"Circuit edges: "
        f"{circuit.number_of_edges():,}"
    )

    experiment_inputs = {}

    for name, types in (
        VISUAL_GROUPS.items()
    ):
        neurons = []

        for cell_type in types:
            neurons.extend(
                selected.get(
                    cell_type,
                    [],
                )
            )

        experiment_inputs[
            name
        ] = list(
            dict.fromkeys(
                neurons
            )
        )

    print()
    print(
        "=" * 72
    )
    print(
        "NORMALIZED VISUAL INPUT COMPARISON"
    )
    print(
        "=" * 72
    )

    results = {}

    for name, input_neurons in (
        experiment_inputs.items()
    ):
        print()
        print(
            f"[{name}]"
        )

        print(
            f"  Inputs: "
            f"{len(input_neurons)}"
        )

        result = run_experiment(
            circuit,
            input_neurons,
        )

        results[name] = result

        first_step = result[
            "first_dn_step"
        ]

        print(
            f"  Total fired: "
            f"{result['total_fired']:,}"
        )

        print(
            f"  Unique DN neurons: "
            f"{result['unique_dn_neurons']:,}"
        )

        print(
            f"  DN firing events: "
            f"{result['dn_total']:,}"
        )

        print(
            f"  First DN step: "
            f"{first_step if first_step is not None else 'none'}"
        )

        print()
        print(
            "  Top DN outputs:"
        )

        print_top_outputs(
            result
        )

    print()
    print(
        "=" * 72
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 72
    )

    print()

    print(
        f"{'Input':<8} "
        f"{'Fired':>10} "
        f"{'Unique DN':>10} "
        f"{'DN events':>10} "
        f"{'First DN':>10}"
    )

    print(
        "-" * 56
    )

    for name, result in (
        results.items()
    ):
        first_step = result[
            "first_dn_step"
        ]

        print(
            f"{name:<8} "
            f"{result['total_fired']:>10,} "
            f"{result['unique_dn_neurons']:>10,} "
            f"{result['dn_total']:>10,} "
            f"{first_step if first_step is not None else '-':>10}"
        )

    print()
    print(
        "Experiment complete."
    )


if __name__ == "__main__":
    main()