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

VISUAL_TYPES = [
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
]

TOP_INPUTS_PER_TYPE = 30
TOP_NEIGHBORS = 8
MAX_HOPS = 5
MIN_SYNAPSES = 5

SIMULATION_STEPS = 12

DECAY = 0.9

THRESHOLDS = [
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
]


def get_neurons_by_type(
    graph: nx.DiGraph,
    cell_type: str,
) -> list[int]:
    return [
        neuron_id
        for neuron_id, data in graph.nodes(
            data=True
        )
        if data.get("cell_type") == cell_type
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

    for cell_type in VISUAL_TYPES:
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

    for _ in range(MAX_HOPS):
        next_frontier = set()

        for neuron_id in frontier:
            if neuron_id in visited:
                continue

            visited.add(neuron_id)

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

    if cell_type.startswith("DN"):
        return cell_type

    return None


def run_experiment(
    circuit: nx.DiGraph,
    input_neurons: list[int],
    threshold: float,
) -> dict[str, int]:
    simulation = NeuralSimulation(
        circuit,
        threshold=threshold,
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

    return dict(dn_firing)


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


def summarize_threshold(
    threshold: float,
    results: dict[str, dict[str, int]],
) -> None:
    total_events = {
        name: sum(outputs.values())
        for name, outputs in results.items()
    }

    active_types = {
        name: len(outputs)
        for name, outputs in results.items()
    }

    pairwise = []

    for name_a, name_b in combinations(
        VISUAL_TYPES,
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

        pairwise.append(
            (
                cosine,
                jaccard,
                name_a,
                name_b,
            )
        )

    pairwise.sort(
        key=lambda row: row[0]
    )

    most_different = pairwise[0]

    t4_t5 = []

    for suffix in ["a", "b", "c", "d"]:
        t4 = f"T4{suffix}"
        t5 = f"T5{suffix}"

        cosine = cosine_similarity(
            results[t4],
            results[t5],
        )

        jaccard = jaccard_similarity(
            results[t4],
            results[t5],
        )

        t4_t5.append(
            (
                cosine,
                jaccard,
                t4,
                t5,
            )
        )

    print()
    print(
        f"=== THRESHOLD {threshold:.1f} ==="
    )

    print(
        "DN active types:"
    )

    for name in VISUAL_TYPES:
        print(
            f"  {name:<4} "
            f"{active_types[name]:>3}"
        )

    print(
        "DN firing events:"
    )

    for name in VISUAL_TYPES:
        print(
            f"  {name:<4} "
            f"{total_events[name]:>5}"
        )

    print()
    print(
        "T4 vs T5 same subtype:"
    )

    for cosine, jaccard, t4, t5 in t4_t5:
        print(
            f"  {t4} vs {t5} "
            f"cosine={cosine:.4f} "
            f"jaccard={jaccard:.4f}"
        )

    print()
    print(
        "Most different visual pair:"
    )

    cosine, jaccard, name_a, name_b = (
        most_different
    )

    print(
        f"  {name_a} vs {name_b} "
        f"cosine={cosine:.4f} "
        f"jaccard={jaccard:.4f}"
    )


def print_summary_table(
    all_results: dict[
        float,
        dict[str, dict[str, int]],
    ],
) -> None:
    print()
    print(
        "============================================================"
    )
    print(
        "THRESHOLD SUMMARY"
    )
    print(
        "============================================================"
    )

    print(
        "threshold | "
        "T4a DN | T5a DN | "
        "T4a events | T5a events | "
        "T4a-T5a cosine | "
        "most different"
    )

    print(
        "----------|--------|--------|"
        "------------|------------|"
        "---------------|----------------"
    )

    for threshold, results in (
        all_results.items()
    ):
        t4a = results["T4a"]
        t5a = results["T5a"]

        cosine = cosine_similarity(
            t4a,
            t5a,
        )

        pairwise = []

        for name_a, name_b in combinations(
            VISUAL_TYPES,
            2,
        ):
            pairwise.append(
                (
                    cosine_similarity(
                        results[name_a],
                        results[name_b],
                    ),
                    name_a,
                    name_b,
                )
            )

        pairwise.sort(
            key=lambda row: row[0]
        )

        min_cosine, name_a, name_b = (
            pairwise[0]
        )

        print(
            f"{threshold:9.1f} | "
            f"{len(t4a):6} | "
            f"{len(t5a):6} | "
            f"{sum(t4a.values()):10} | "
            f"{sum(t5a.values()):10} | "
            f"{cosine:13.4f} | "
            f"{name_a} vs {name_b} "
            f"({min_cosine:.4f})"
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

    print()
    print(
        "Running threshold sweep..."
    )

    experiment_inputs = {
        cell_type: selected[cell_type]
        for cell_type in VISUAL_TYPES
    }

    all_results = {}

    for threshold in THRESHOLDS:
        print()
        print(
            f"Running threshold={threshold:.1f}"
        )

        results = {}

        for name in VISUAL_TYPES:
            outputs = run_experiment(
                circuit,
                experiment_inputs[name],
                threshold,
            )

            results[name] = outputs

            print(
                f"  {name:<4} "
                f"DN={len(outputs):>3} "
                f"events={sum(outputs.values()):>5}"
            )

        all_results[
            threshold
        ] = results

        summarize_threshold(
            threshold,
            results,
        )

    print_summary_table(
        all_results
    )

    print()
    print(
        "Experiment complete."
    )


if __name__ == "__main__":
    main()