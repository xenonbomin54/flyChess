from __future__ import annotations

import pandas as pd
import networkx as nx

from src.brain.simulation import NeuralSimulation
from src.connectome.graph import build_graph


CONNECTIONS_PATH = "data/raw/connections_princeton.csv.gz"
ANNOTATIONS_PATH = "data/raw/neuron_annotations.tsv"

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

CANDIDATE_DNS = [
    "DNa02",
    "DNa03",
    "DNa09",
    "DNa10",
    "DNa11",
    "DNa13",
    "DNa16",
    "DNae002",
    "DNae003",
    "DNae010",
    "DNb01",
    "DNb03",
    "DNbe001",
    "DNg04",
    "DNg05_a",
    "DNg110",
    "DNg41",
    "DNg42",
    "DNg46",
    "DNg71",
    "DNg78",
    "DNge006",
    "DNge026",
    "DNge037",
    "DNge043",
    "DNge045",
    "DNge086",
    "DNp15",
    "DNp18",
    "DNp22",
    "DNp26",
    "DNp31",
    "DNp51",
    "DNpe019",
]

N_INPUTS_PER_TYPE = 30

TOP_K = 8
MIN_SYNAPSES = 5
MAX_HOPS = 5

THRESHOLD = 0.3
DECAY = 0.9
STEPS = 12


def load_annotations(path: str) -> pd.DataFrame:
    annotations = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    annotations["root_id"] = pd.to_numeric(
        annotations["root_id"],
        errors="coerce",
    )

    annotations["cell_type"] = (
        annotations["cell_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return annotations


def select_visual_neurons(
    annotations: pd.DataFrame,
) -> dict[str, list[int]]:
    result = {}

    for visual_type in VISUAL_TYPES:
        ids = (
            annotations.loc[
                annotations["cell_type"] == visual_type,
                "root_id",
            ]
            .dropna()
            .astype("int64")
            .drop_duplicates()
            .tolist()
        )

        result[visual_type] = ids[:N_INPUTS_PER_TYPE]

    return result


def select_candidate_dns(
    annotations: pd.DataFrame,
) -> dict[str, list[int]]:
    result = {}

    for dn_type in CANDIDATE_DNS:
        ids = (
            annotations.loc[
                annotations["cell_type"] == dn_type,
                "root_id",
            ]
            .dropna()
            .astype("int64")
            .drop_duplicates()
            .tolist()
        )

        if ids:
            result[dn_type] = ids

    return result


def get_top_neighbors(
    graph: nx.DiGraph,
    neuron_id: int,
) -> list[int]:
    if neuron_id not in graph:
        return []

    candidates = []

    for target in graph.successors(neuron_id):
        synapses = int(
            graph[neuron_id][target].get(
                "syn_count",
                0,
            )
        )

        if synapses < MIN_SYNAPSES:
            continue

        candidates.append(
            (
                target,
                synapses,
            )
        )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        target
        for target, _ in candidates[:TOP_K]
    ]


def build_visual_circuit(
    graph: nx.DiGraph,
    visual_neurons: dict[str, list[int]],
) -> nx.DiGraph:
    """
    Build a bounded visual circuit.

    Starting from the selected T4/T5 neurons, each hop
    follows only the top-K outgoing connections.

    A neuron is expanded only once. This keeps the circuit
    bounded and prevents cycles from repeatedly expanding.
    """

    initial_nodes = set()

    for neurons in visual_neurons.values():
        for neuron_id in neurons:
            if neuron_id in graph:
                initial_nodes.add(neuron_id)

    circuit = nx.DiGraph()

    circuit.add_nodes_from(
        initial_nodes
    )

    seen = set(initial_nodes)
    frontier = set(initial_nodes)

    print(
        f"Initial frontier: "
        f"{len(frontier):,}"
    )

    for hop in range(1, MAX_HOPS + 1):
        next_frontier = set()
        added_edges = 0

        for source in frontier:
            neighbors = get_top_neighbors(
                graph,
                source,
            )

            for target in neighbors:
                data = graph[source][target]

                circuit.add_edge(
                    source,
                    target,
                    **data,
                )

                added_edges += 1

                if target not in seen:
                    seen.add(target)
                    next_frontier.add(target)

        frontier = next_frontier

        print(
            f"  hop {hop}: "
            f"new_nodes={len(frontier):,} "
            f"total_nodes={circuit.number_of_nodes():,} "
            f"total_edges={circuit.number_of_edges():,}"
        )

        if not frontier:
            break

    return circuit


def run_simulation(
    circuit: nx.DiGraph,
    visual_neurons: dict[str, list[int]],
    candidate_dns: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    results = {}

    for visual_type in VISUAL_TYPES:
        print()
        print("=" * 70)
        print(f"SIMULATING {visual_type}")
        print("=" * 70)

        simulation = NeuralSimulation(
            circuit,
            threshold=THRESHOLD,
            decay=DECAY,
        )

        inputs = [
            neuron_id
            for neuron_id in visual_neurons[visual_type]
            if neuron_id in circuit
        ]

        simulation.stimulate_many(
            inputs,
            signal=1.0,
        )

        history = simulation.run(
            steps=STEPS,
        )

        dn_counts = {}

        fired_sets = [
            set(step)
            for step in history
        ]

        for dn_type, neuron_ids in candidate_dns.items():
            count = 0

            for fired in fired_sets:
                count += sum(
                    neuron_id in fired
                    for neuron_id in neuron_ids
                )

            dn_counts[dn_type] = count

        results[visual_type] = dn_counts

        total_events = sum(
            len(step)
            for step in history
        )

        active = {
            dn_type: count
            for dn_type, count in dn_counts.items()
            if count > 0
        }

        print(
            f"total firing events: "
            f"{total_events}"
        )

        print(
            f"active candidate DNs: "
            f"{len(active)}"
        )

        for dn_type, count in sorted(
            active.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"  {dn_type:10s} {count:4d}"
            )

    return results


def print_summary(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print("=" * 70)
    print("CROSS-VISUAL DN FIRING")
    print("=" * 70)

    all_dns = sorted(
        {
            dn_type
            for visual_result in results.values()
            for dn_type in visual_result
        }
    )

    print(
        f"{'DN':10s}"
        f"{'visuals':>8s}"
        f"{'total':>8s}"
        f"{'max':>8s}"
    )

    for dn_type in all_dns:
        values = [
            results[visual_type].get(
                dn_type,
                0,
            )
            for visual_type in VISUAL_TYPES
        ]

        active = sum(
            value > 0
            for value in values
        )

        total = sum(values)
        maximum = max(values)

        if total == 0:
            continue

        print(
            f"{dn_type:10s}"
            f"{active:8d}"
            f"{total:8d}"
            f"{maximum:8d}"
        )


def print_matrix(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print("=" * 70)
    print("VISUAL × DN FIRING MATRIX")
    print("=" * 70)

    active_dns = sorted(
        {
            dn_type
            for visual_result in results.values()
            for dn_type, count in visual_result.items()
            if count > 0
        }
    )

    if not active_dns:
        print("No candidate DN fired.")
        return

    print(
        "DN".ljust(12)
        + "".join(
            visual_type.rjust(8)
            for visual_type in VISUAL_TYPES
        )
    )

    for dn_type in active_dns:
        values = [
            results[visual_type].get(
                dn_type,
                0,
            )
            for visual_type in VISUAL_TYPES
        ]

        print(
            dn_type.ljust(12)
            + "".join(
                str(value).rjust(8)
                for value in values
            )
        )


def print_discriminative(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print("=" * 70)
    print("DISCRIMINATIVE DN FIRING")
    print("=" * 70)

    all_dns = sorted(
        {
            dn_type
            for visual_result in results.values()
            for dn_type in visual_result
        }
    )

    for visual_type in VISUAL_TYPES:
        scores = []

        for dn_type in all_dns:
            own = results[visual_type].get(
                dn_type,
                0,
            )

            if own <= 0:
                continue

            others = [
                results[other].get(
                    dn_type,
                    0,
                )
                for other in VISUAL_TYPES
                if other != visual_type
            ]

            mean_other = (
                sum(others) / len(others)
                if others
                else 0.0
            )

            score = own / (
                1.0 + mean_other
            )

            scores.append(
                (
                    score,
                    dn_type,
                    own,
                    mean_other,
                )
            )

        scores.sort(
            reverse=True
        )

        print()
        print(f"[{visual_type}]")

        for score, dn_type, own, mean_other in scores[:10]:
            print(
                f"  {dn_type:10s}"
                f" score={score:.3f}"
                f" own={own}"
                f" other_mean={mean_other:.2f}"
            )


def save_results(
    results: dict[str, dict[str, int]],
    path: str,
) -> None:
    rows = []

    for visual_type, dn_counts in results.items():
        for dn_type, firing_count in dn_counts.items():
            rows.append(
                {
                    "visual_type": visual_type,
                    "dn_type": dn_type,
                    "firing_count": firing_count,
                }
            )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def main() -> None:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    connections = pd.read_csv(
        CONNECTIONS_PATH
    )

    annotations = load_annotations(
        ANNOTATIONS_PATH
    )

    print(
        f"Connection rows: "
        f"{len(connections):,}"
    )

    print(
        f"Annotation rows: "
        f"{len(annotations):,}"
    )

    print()
    print("=" * 70)
    print("BUILDING GRAPH")
    print("=" * 70)

    graph = build_graph(
        connections
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges():,}"
    )

    visual_neurons = select_visual_neurons(
        annotations
    )

    candidate_dns = select_candidate_dns(
        annotations
    )

    print()
    print("=" * 70)
    print("SELECTED NEURONS")
    print("=" * 70)

    for visual_type, neurons in visual_neurons.items():
        print(
            f"{visual_type}: "
            f"{len(neurons)}"
        )

    print(
        f"Candidate DN types: "
        f"{len(candidate_dns)}"
    )

    print()
    print("=" * 70)
    print("BUILDING VISUAL CIRCUIT")
    print("=" * 70)

    circuit = build_visual_circuit(
        graph,
        visual_neurons,
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
    print("=" * 70)
    print("RUNNING NEURAL SIMULATION")
    print("=" * 70)

    results = run_simulation(
        circuit,
        visual_neurons,
        candidate_dns,
    )

    print_summary(
        results
    )

    print_matrix(
        results
    )

    print_discriminative(
        results
    )

    output_path = (
        "data/processed/"
        "visual_dn_firing.csv"
    )

    save_results(
        results,
        output_path,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()