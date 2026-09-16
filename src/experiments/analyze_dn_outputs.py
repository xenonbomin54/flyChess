from __future__ import annotations

from collections import defaultdict
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

THRESHOLD = 0.3
DECAY = 0.9

TOP_OUTPUTS = 20


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
                for _, synapses in outgoing
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

        selected[cell_type] = [
            neuron_id
            for neuron_id, _ in candidates[
                :TOP_INPUTS_PER_TYPE
            ]
        ]

    return selected


def build_circuit(
    graph: nx.DiGraph,
    input_neurons: list[int],
) -> nx.DiGraph:
    circuit = nx.DiGraph()

    frontier = set(input_neurons)
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
                circuit.add_node(
                    target,
                    cell_type=graph.nodes[
                        target
                    ].get(
                        "cell_type",
                        "Unknown",
                    ),
                )

                circuit.add_edge(
                    neuron_id,
                    target,
                    syn_count=synapses,
                )

                next_frontier.add(target)

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

            dn_firing[cell_type] += 1

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


def print_top_outputs(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== TOP DN OUTPUTS ==="
    )

    for name in VISUAL_TYPES:
        outputs = results[name]

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
            ranked[:TOP_OUTPUTS],
            start=1,
        ):
            print(
                f"  {index:2}. "
                f"{cell_type:<10} "
                f"{count:>4}"
            )


def build_presence(
    results: dict[str, dict[str, int]],
) -> dict[str, set[str]]:
    return {
        name: set(outputs.keys())
        for name, outputs in results.items()
    }


def print_common_outputs(
    results: dict[str, dict[str, int]],
) -> None:
    presence = build_presence(results)

    all_sets = list(
        presence.values()
    )

    common = set.intersection(
        *all_sets
    )

    print()
    print(
        "=== COMMON DN OUTPUTS ==="
    )

    print(
        f"DN types active in all "
        f"8 visual inputs: {len(common)}"
    )

    ranked = []

    for cell_type in common:
        values = [
            results[name].get(
                cell_type,
                0,
            )
            for name in VISUAL_TYPES
        ]

        mean = sum(values) / len(values)

        spread = max(values) - min(values)

        ranked.append(
            (
                mean,
                spread,
                cell_type,
                values,
            )
        )

    ranked.sort(
        reverse=True
    )

    for mean, spread, cell_type, values in ranked[
        :TOP_OUTPUTS
    ]:
        value_text = " ".join(
            f"{value:>3}"
            for value in values
        )

        print(
            f"{cell_type:<10} "
            f"mean={mean:6.2f} "
            f"spread={spread:>2} "
            f"[{value_text}]"
        )


def print_unique_outputs(
    results: dict[str, dict[str, int]],
) -> None:
    presence = build_presence(results)

    print()
    print(
        "=== RARE / DISCRIMINATIVE DN OUTPUTS ==="
    )

    for name in VISUAL_TYPES:
        own = presence[name]

        others = set.union(
            *[
                presence[other]
                for other in VISUAL_TYPES
                if other != name
            ]
        )

        unique = own - others

        print()
        print(
            f"[{name}] "
            f"unique DN types={len(unique)}"
        )

        ranked = sorted(
            [
                (
                    results[name][cell_type],
                    cell_type,
                )
                for cell_type in unique
            ],
            reverse=True,
        )

        if not ranked:
            print(
                "  none"
            )
            continue

        for count, cell_type in ranked[
            :TOP_OUTPUTS
        ]:
            print(
                f"  {cell_type:<10} "
                f"{count:>4}"
            )


def print_discriminative_scores(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== DISCRIMINATIVE OUTPUT SCORE ==="
    )

    print(
        "Score = own firing count / "
        "(1 + mean firing count in other inputs)"
    )

    for name in VISUAL_TYPES:
        scores = []

        for cell_type, own_count in (
            results[name].items()
        ):
            other_values = [
                results[other].get(
                    cell_type,
                    0,
                )
                for other in VISUAL_TYPES
                if other != name
            ]

            mean_other = (
                sum(other_values)
                / len(other_values)
            )

            score = own_count / (
                1.0 + mean_other
            )

            scores.append(
                (
                    score,
                    own_count,
                    mean_other,
                    cell_type,
                )
            )

        scores.sort(
            reverse=True
        )

        print()
        print(
            f"[{name}]"
        )

        for (
            score,
            own_count,
            mean_other,
            cell_type,
        ) in scores[:TOP_OUTPUTS]:
            print(
                f"  {cell_type:<10} "
                f"score={score:6.2f} "
                f"own={own_count:>3} "
                f"other_mean={mean_other:6.2f}"
            )


def print_pairwise_similarity(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== PAIRWISE SIMILARITY ==="
    )

    for i, name_a in enumerate(
        VISUAL_TYPES
    ):
        for name_b in VISUAL_TYPES[
            i + 1:
        ]:
            similarity = cosine_similarity(
                results[name_a],
                results[name_b],
            )

            print(
                f"{name_a:<4} vs "
                f"{name_b:<4} "
                f"{similarity:.4f}"
            )


def print_subtype_comparison(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print(
        "=== T4 / T5 SUBTYPE COMPARISON ==="
    )

    for suffix in "abcd":
        t4 = f"T4{suffix}"
        t5 = f"T5{suffix}"

        a = results[t4]
        b = results[t5]

        keys = set(a) | set(b)

        differences = []

        for cell_type in keys:
            difference = abs(
                a.get(cell_type, 0)
                - b.get(cell_type, 0)
            )

            if difference > 0:
                differences.append(
                    (
                        difference,
                        cell_type,
                        a.get(
                            cell_type,
                            0,
                        ),
                        b.get(
                            cell_type,
                            0,
                        ),
                    )
                )

        differences.sort(
            reverse=True
        )

        print()
        print(
            f"{t4} vs {t5}"
        )

        print(
            f"cosine="
            f"{cosine_similarity(a, b):.4f}"
        )

        for (
            difference,
            cell_type,
            t4_count,
            t5_count,
        ) in differences[:10]:
            print(
                f"  {cell_type:<10} "
                f"T4={t4_count:>3} "
                f"T5={t5_count:>3} "
                f"diff={difference:>3}"
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
        f"Running DN analysis "
        f"(threshold={THRESHOLD})..."
    )

    results = {}

    for name in VISUAL_TYPES:
        print(
            f"  {name:<4}",
            end="",
            flush=True,
        )

        outputs = run_experiment(
            circuit,
            selected[name],
        )

        results[name] = outputs

        print(
            f" DN={len(outputs):>3} "
            f"events={sum(outputs.values()):>4}"
        )

    print_top_outputs(
        results
    )

    print_common_outputs(
        results
    )

    print_unique_outputs(
        results
    )

    print_discriminative_scores(
        results
    )

    print_subtype_comparison(
        results
    )

    print_pairwise_similarity(
        results
    )

    print()
    print(
        "Experiment complete."
    )


if __name__ == "__main__":
    main()