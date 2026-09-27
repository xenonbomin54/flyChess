from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.brain.simulation import NeuralSimulation
from src.brain.visual_input import FlappyBirdState, VisualEncoder
from src.connectome.graph import build_graph
from src.connectome.loader import load_connectome


CONNECTOME_PATH = "data/raw/connections_princeton.csv.gz"
ANNOTATION_PATH = "data/raw/neuron_annotations.tsv"

VISUAL_TYPES = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)

CANDIDATE_DN_TYPES = {
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
    "DNge043",
    "DNp15",
    "DNp18",
    "DNp26",
    "DNp31",
    "DNp51",
    "DNpe019",
}

NEURONS_PER_VISUAL_TYPE = 30
TOP_K_CONNECTIONS = 8
MIN_SYNAPSES = 5
MAX_HOPS = 5

THRESHOLD = 0.3
DECAY = 0.9
STEPS = 20


def load_annotations(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def get_visual_neurons(
    annotations: pd.DataFrame,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}

    for visual_type in VISUAL_TYPES:
        rows = annotations[
            annotations["cell_type"].astype(str) == visual_type
        ]

        neuron_ids = (
            rows["root_id"]
            .dropna()
            .astype("int64")
            .tolist()
        )

        result[visual_type] = neuron_ids[:NEURONS_PER_VISUAL_TYPE]

    return result


def get_dn_neurons(
    annotations: pd.DataFrame,
) -> dict[int, str]:
    result: dict[int, str] = {}

    for row in annotations.itertuples(index=False):
        cell_type = str(getattr(row, "cell_type", ""))

        if cell_type not in CANDIDATE_DN_TYPES:
            continue

        try:
            root_id = int(row.root_id)
        except (TypeError, ValueError):
            continue

        result[root_id] = cell_type

    return result


def build_visual_circuit(
    graph,
    visual_neurons: dict[str, list[int]],
    dn_neurons: dict[int, str],
):
    """
    Build a bounded circuit starting from T4/T5 neurons.

    Only the strongest outgoing connections are retained.
    The circuit is expanded for a limited number of hops.
    """

    selected_edges: dict[tuple[int, int], int] = {}
    visited: set[int] = set()

    frontier: list[int] = []

    for neuron_ids in visual_neurons.values():
        frontier.extend(neuron_ids)

    frontier = list(dict.fromkeys(frontier))

    print()
    print("=" * 70)
    print("BUILDING FLAPPY VISUAL CIRCUIT")
    print("=" * 70)
    print(f"Initial visual neurons: {len(frontier)}")

    for hop in range(1, MAX_HOPS + 1):
        next_frontier: list[int] = []
        new_nodes = 0

        for source in frontier:
            if source in visited:
                continue

            visited.add(source)

            if source not in graph:
                continue

            candidates = []

            for target in graph.successors(source):
                data = graph[source][target]

                syn_count = int(
                    data.get("syn_count", 0)
                )

                if syn_count < MIN_SYNAPSES:
                    continue

                candidates.append(
                    (
                        target,
                        syn_count,
                    )
                )

            candidates.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            for target, syn_count in candidates[:TOP_K_CONNECTIONS]:
                selected_edges[
                    (source, target)
                ] = syn_count

                if target not in visited:
                    next_frontier.append(target)
                    new_nodes += 1

        frontier = list(dict.fromkeys(next_frontier))

        print(
            f"hop {hop}: "
            f"new_nodes={new_nodes} "
            f"total_nodes={len(visited | set(frontier))} "
            f"total_edges={len(selected_edges)}"
        )

        if not frontier:
            break

    import networkx as nx

    circuit = nx.DiGraph()

    for (source, target), syn_count in selected_edges.items():
        circuit.add_edge(
            source,
            target,
            syn_count=syn_count,
        )

    return circuit


def convert_visual_activity_to_signal(
    activity: float,
) -> float:
    if activity <= 0:
        return 0.0

    return 0.3 + (
        0.7 * min(
            activity,
            1.0,
        )
    )


def print_visual_input(
    features: dict[str, float],
) -> None:
    print()
    print("=" * 70)
    print("FLAPPY BIRD VISUAL INPUT")
    print("=" * 70)

    for visual_type in VISUAL_TYPES:
        print(
            f"{visual_type}: "
            f"{features.get(visual_type, 0.0):.4f}"
        )


def main() -> None:
    print("=" * 70)
    print("LOADING CONNECTOME")
    print("=" * 70)

    df = load_connectome(
        CONNECTOME_PATH
    )

    print(
        f"Connection rows: {len(df):,}"
    )

    graph = build_graph(df)

    print(
        f"Nodes: {graph.number_of_nodes():,}"
    )

    print(
        f"Edges: {graph.number_of_edges():,}"
    )

    del df

    annotations = load_annotations(
        ANNOTATION_PATH
    )

    visual_neurons = get_visual_neurons(
        annotations
    )

    dn_neurons = get_dn_neurons(
        annotations
    )

    print()
    print("=" * 70)
    print("VISUAL NEURON COUNTS")
    print("=" * 70)

    for visual_type in VISUAL_TYPES:
        print(
            f"{visual_type}: "
            f"{len(visual_neurons[visual_type])}"
        )

    print(
        f"Candidate DN neurons: "
        f"{len(dn_neurons):,}"
    )

    circuit = build_visual_circuit(
        graph,
        visual_neurons,
        dn_neurons,
    )

    circuit_dn_neurons = {
        neuron_id: cell_type
        for neuron_id, cell_type in dn_neurons.items()
        if neuron_id in circuit
    }

    print()
    print("=" * 70)
    print("CIRCUIT")
    print("=" * 70)
    print(
        f"Nodes: {circuit.number_of_nodes():,}"
    )
    print(
        f"Edges: {circuit.number_of_edges():,}"
    )
    print(
        f"DN candidates inside circuit: "
        f"{len(circuit_dn_neurons):,}"
    )

    if circuit_dn_neurons:
        dn_type_counts = defaultdict(int)

        for cell_type in circuit_dn_neurons.values():
            dn_type_counts[cell_type] += 1

        print()
        print("DN TYPES REACHED:")

        for cell_type, count in sorted(
            dn_type_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"{cell_type:<12} {count}"
            )

    state = FlappyBirdState(
        bird_x=100.0,
        bird_y=300.0,
        pipe_x=300.0,
        pipe_gap_y=350.0,
        pipe_width=80.0,
        pipe_gap_height=150.0,
        screen_width=800.0,
        screen_height=600.0,
    )

    encoder = VisualEncoder()

    features = encoder.encode(
        state
    )

    print_visual_input(
        features
    )

    simulation = NeuralSimulation(
        circuit,
        threshold=THRESHOLD,
        decay=DECAY,
    )

    print()
    print("=" * 70)
    print("STIMULATED VISUAL NEURONS")
    print("=" * 70)

    for visual_type in VISUAL_TYPES:
        activity = features.get(
            visual_type,
            0.0,
        )

        signal = convert_visual_activity_to_signal(
            activity
        )

        if signal <= 0:
            continue

        neuron_ids = visual_neurons[
            visual_type
        ]

        circuit_neurons = [
            neuron_id
            for neuron_id in neuron_ids
            if neuron_id in circuit
        ]

        for neuron_id in circuit_neurons:
            simulation.stimulate(
                neuron_id,
                signal,
            )

        print(
            f"{visual_type}: "
            f"signal={signal:.4f} "
            f"({len(circuit_neurons)} neurons)"
        )

    print()
    print("=" * 70)
    print("NEURAL SIMULATION")
    print("=" * 70)

    history = []

    for step in range(STEPS):
        fired = simulation.step()
        history.append(fired)

        print(
            f"Step {step:02d}: "
            f"{len(fired)} neurons fired"
        )

        dn_fired = [
            neuron_id
            for neuron_id in fired
            if neuron_id in circuit_dn_neurons
        ]

        if dn_fired:
            print(
                "  >>> DN FIRING:"
            )

            dn_counts = defaultdict(int)

            for neuron_id in dn_fired:
                dn_counts[
                    circuit_dn_neurons[neuron_id]
                ] += 1

            for cell_type, count in sorted(
                dn_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                print(
                    f"      {cell_type:<12} "
                    f"{count}"
                )

    print()
    print("=" * 70)
    print("TOP NEURAL OUTPUTS")
    print("=" * 70)

    counts: dict[int, int] = defaultdict(int)

    for fired in history:
        for neuron_id in fired:
            counts[neuron_id] += 1

    top_outputs = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:30]

    type_by_id = {}

    for row in annotations.itertuples(index=False):
        try:
            root_id = int(row.root_id)
        except (TypeError, ValueError):
            continue

        type_by_id[root_id] = str(
            getattr(
                row,
                "cell_type",
                "Unknown",
            )
        )

    for neuron_id, count in top_outputs:
        print(
            f"{type_by_id.get(neuron_id, 'Unknown'):<16} "
            f"{count}"
        )

    print()
    print("=" * 70)
    print("DN OUTPUT SUMMARY")
    print("=" * 70)

    dn_total_counts: dict[str, int] = defaultdict(int)

    for fired in history:
        for neuron_id in fired:
            if neuron_id in circuit_dn_neurons:
                dn_total_counts[
                    circuit_dn_neurons[neuron_id]
                ] += 1

    if not dn_total_counts:
        print(
            "No candidate DN fired."
        )
        print(
            "The visual circuit reached intermediate neurons "
            "but did not produce DN firing within the simulation window."
        )
    else:
        for cell_type, count in sorted(
            dn_total_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"{cell_type:<16} "
                f"{count}"
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
