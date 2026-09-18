from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pandas as pd

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

N_VISUAL_NEURONS = 30
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

        if len(ids) < N_VISUAL_NEURONS:
            raise ValueError(
                f"{visual_type}: "
                f"{len(ids)} neurons found, "
                f"need {N_VISUAL_NEURONS}"
            )

        result[visual_type] = ids[:N_VISUAL_NEURONS]

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


def build_visual_circuit(
    graph: nx.DiGraph,
    visual_neurons: dict[str, list[int]],
) -> nx.DiGraph:
    """Build the same bounded visual circuit used in strength analysis."""

    selected = set()

    for neurons in visual_neurons.values():
        selected.update(neurons)

    frontier = set(selected)

    for _ in range(MAX_HOPS):
        next_frontier = set()

        for neuron_id in frontier:
            if neuron_id not in graph:
                continue

            neighbors = []

            for target in graph.successors(neuron_id):
                synapses = int(
                    graph[neuron_id][target].get(
                        "syn_count",
                        0,
                    )
                )

                if synapses < MIN_SYNAPSES:
                    continue

                neighbors.append(
                    (target, synapses)
                )

            neighbors.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            for target, _ in neighbors[:TOP_K]:
                if target not in selected:
                    next_frontier.add(target)

        selected.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return graph.subgraph(selected).copy()


def run_visual_simulation(
    circuit: nx.DiGraph,
    visual_neurons: dict[str, list[int]],
    candidate_dns: dict[str, list[int]],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, list[list[int]]],
]:
    results = {}
    histories = {}

    for visual_type, inputs in visual_neurons.items():
        print()
        print("=" * 70)
        print(f"SIMULATING {visual_type}")
        print("=" * 70)

        simulation = NeuralSimulation(
            circuit,
            threshold=THRESHOLD,
            decay=DECAY,
        )

        simulation.stimulate_many(
            inputs,
            signal=1.0,
        )

        history = simulation.run(
            steps=STEPS,
        )

        histories[visual_type] = history

        dn_counts = {}

        for dn_type, neuron_ids in candidate_dns.items():
            count = 0

            for fired in history:
                fired_set = set(fired)

                for neuron_id in neuron_ids:
                    if neuron_id in fired_set:
                        count += 1

            dn_counts[dn_type] = count

        results[visual_type] = dn_counts

        total_fired = sum(
            len(step)
            for step in history
        )

        active_dns = {
            dn_type: count
            for dn_type, count in dn_counts.items()
            if count > 0
        }

        print(
            f"total firing events: {total_fired}"
        )
        print(
            f"active candidate DNs: "
            f"{len(active_dns)}"
        )

        for dn_type, count in sorted(
            active_dns.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:15]:
            print(
                f"  {dn_type:10s} "
                f"{count:4d}"
            )

    return results, histories


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

    df = pd.DataFrame(rows)

    df.to_csv(
        path,
        index=False,
    )


def print_cross_visual_summary(
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

    rows = []

    for dn_type in all_dns:
        values = [
            results[visual_type].get(
                dn_type,
                0,
            )
            for visual_type in VISUAL_TYPES
        ]

        active_visuals = sum(
            value > 0
            for value in values
        )

        total = sum(values)
        maximum = max(values)

        rows.append(
            (
                dn_type,
                active_visuals,
                total,
                maximum,
                values,
            )
        )

    rows.sort(
        key=lambda row: (
            row[1],
            row[2],
            row[3],
        ),
        reverse=True,
    )

    print(
        f"{'DN':10s} "
        f"{'visuals':>7s} "
        f"{'total':>7s} "
        f"{'max':>7s}"
    )

    for dn_type, active, total, maximum, _ in rows[:30]:
        print(
            f"{dn_type:10s} "
            f"{active:7d} "
            f"{total:7d} "
            f"{maximum:7d}"
        )


def print_visual_matrix(
    results: dict[str, dict[str, int]],
) -> None:
    print()
    print("=" * 70)
    print("VISUAL × DN FIRING MATRIX")
    print("=" * 70)

    interesting_dns = sorted(
        {
            dn_type
            for visual_result in results.values()
            for dn_type, count in visual_result.items()
            if count > 0
        }
    )

    print(
        "DN".ljust(12)
        + "".join(
            visual_type.rjust(8)
            for visual_type in VISUAL_TYPES
        )
    )

    for dn_type in interesting_dns:
        values = []

        for visual_type in VISUAL_TYPES:
            values.append(
                results[visual_type].get(
                    dn_type,
                    0,
                )
            )

        print(
            dn_type.ljust(12)
            + "".join(
                str(value).rjust(8)
                for value in values
            )
        )


def print_discriminative_outputs(
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

            others = [
                results[other].get(
                    dn_type,
                    0,
                )
                for other in VISUAL_TYPES
                if other != visual_type
            ]

            if own <= 0:
                continue

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
                f"  {dn_type:10s} "
                f"score={score:.3f} "
                f"own={own:3d} "
                f"other_mean={mean_other:.2f}"
            )


def main() -> None:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    connections = pd.read_csv(
        CONNECTIONS_PATH
    )

    print(
        f"Connection rows: "
        f"{len(connections):,}"
    )

    annotations = load_annotations(
        ANNOTATIONS_PATH
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
        f"Nodes: {graph.number_of_nodes():,}"
    )
    print(
        f"Edges: {graph.number_of_edges():,}"
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

    results, _ = run_visual_simulation(
        circuit,
        visual_neurons,
        candidate_dns,
    )

    print_cross_visual_summary(
        results
    )

    print_visual_matrix(
        results
    )

    print_discriminative_outputs(
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