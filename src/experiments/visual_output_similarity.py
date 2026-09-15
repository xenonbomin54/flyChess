from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import sqrt

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
) -> dict[str, int]:
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

    dn_firing = defaultdict(int)

    for fired in history:
        for neuron_id in fired:
            cell_type = get_dn_type(
                circuit,
                neuron_id,
            )

            if cell_type is None:
                continue

            dn_firing[
                cell_type
            ] += 1

    return dict(
        dn_firing
    )


def cosine_similarity(
    a: dict[str, int],
    b: dict[str, int],
) -> float:
    keys = set(a) | set(b)

    if not keys:
        return 0.0

    dot = sum(
        a.get(key, 0)
        * b.get(key, 0)
        for key in keys
    )

    norm_a = sqrt(
        sum(
            value * value
            for value in a.values()
        )
    )

    norm_b = sqrt(
        sum(
            value * value
            for value in b.values()
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (
        norm_a * norm_b
    )


def jaccard_similarity(
    a: dict[str, int],
    b: dict[str, int],
) -> float:
    active_a = {
        key
        for key, value in a.items()
        if value > 0
    }

    active_b = {
        key
        for key, value in b.items()
        if value > 0
    }

    union = active_a | active_b

    if not union:
        return 0.0

    intersection = (
        active_a & active_b
    )

    return len(intersection) / len(
        union
    )


def print_pairwise_similarity(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== PAIRWISE OUTPUT SIMILARITY ==="
    )

    rows = []

    names = list(
        results.keys()
    )

    for name_a, name_b in combinations(
        names,
        2,
    ):
        cosine = cosine_similarity(
            results[name_a],
            results[name_b],
        )

        jaccard = jaccard_similarity(
            results[name_a],
            results[name_b],
        )

        rows.append(
            (
                cosine,
                jaccard,
                name_a,
                name_b,
            )
        )

    rows.sort(
        key=lambda row: row[0]
    )

    print()
    print(
        "Most different:"
    )

    for cosine, jaccard, name_a, name_b in rows[
        :10
    ]:
        print(
            f"  {name_a:<8} vs "
            f"{name_b:<8} "
            f"cosine={cosine:.4f} "
            f"jaccard={jaccard:.4f}"
        )

    print()
    print(
        "Most similar:"
    )

    for cosine, jaccard, name_a, name_b in rows[
        -10:
    ][::-1]:
        print(
            f"  {name_a:<8} vs "
            f"{name_b:<8} "
            f"cosine={cosine:.4f} "
            f"jaccard={jaccard:.4f}"
        )


def print_top_outputs(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== TOP OUTPUTS ==="
    )

    for name, outputs in results.items():
        ranked = sorted(
            outputs.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        print()
        print(
            f"[{name}]"
        )

        for index, (
            cell_type,
            count,
        ) in enumerate(
            ranked[:10],
            start=1,
        ):
            print(
                f"  {index:2}. "
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
        "Selecting visual inputs..."
    )

    selected = select_input_neurons(
        graph
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

    print(
        f"Total selected inputs: "
        f"{len(all_inputs)}"
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
        "Running experiments..."
    )

    results = {}

    for name, input_neurons in (
        experiment_inputs.items()
    ):
        print(
            f"  {name:<8} "
            f"inputs={len(input_neurons):>3}",
            end="",
            flush=True,
        )

        outputs = run_experiment(
            circuit,
            input_neurons,
        )

        results[
            name
        ] = outputs

        print(
            f"  DN types={len(outputs):>3}"
        )

    print_top_outputs(
        results
    )

    print_pairwise_similarity(
        results
    )

    print()
    print(
        "=== GROUP COMPARISON ==="
    )

    group_pairs = [
        ("T4a", "T4b"),
        ("T4a", "T4c"),
        ("T4a", "T4d"),
        ("T5a", "T5b"),
        ("T5a", "T5c"),
        ("T5a", "T5d"),
        ("T4a", "T5a"),
        ("T4b", "T5b"),
        ("T4c", "T5c"),
        ("T4d", "T5d"),
    ]

    print()

    for name_a, name_b in group_pairs:
        cosine = cosine_similarity(
            results[name_a],
            results[name_b],
        )

        jaccard = jaccard_similarity(
            results[name_a],
            results[name_b],
        )

        print(
            f"{name_a:<4} vs "
            f"{name_b:<4} "
            f"cosine={cosine:.4f} "
            f"jaccard={jaccard:.4f}"
        )

    print()
    print(
        "Experiment complete."
    )


if __name__ == "__main__":
    main()