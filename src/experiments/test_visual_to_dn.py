from __future__ import annotations

from collections import defaultdict

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

TARGET_TYPES = {
    "DNp04",
    "DNp103",
    "DNb05",
}

MIN_SYNAPSES = 100
MAX_HOPS = 5
TOP_K_PER_NODE = 30


def build_cell_type_network(
    graph,
) -> dict[str, list[tuple[str, int]]]:
    """Build a strong cell-type adjacency network."""

    connections = defaultdict(
        lambda: defaultdict(int)
    )

    for source, target, data in graph.edges(
        data=True
    ):
        source_type = graph.nodes[
            source
        ].get("cell_type", "Unknown")

        target_type = graph.nodes[
            target
        ].get("cell_type", "Unknown")

        if (
            source_type == "Unknown"
            or target_type == "Unknown"
        ):
            continue

        syn_count = int(
            data.get("syn_count", 0)
        )

        connections[
            source_type
        ][target_type] += syn_count

    adjacency = {}

    for source_type, targets in connections.items():
        filtered = [
            (
                target_type,
                syn_count,
            )
            for target_type, syn_count
            in targets.items()
            if syn_count >= MIN_SYNAPSES
        ]

        filtered.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        adjacency[source_type] = (
            filtered[:TOP_K_PER_NODE]
        )

    return adjacency


def find_paths(
    start_type: str,
    target_types: set[str],
    adjacency,
) -> list[dict]:
    """Find strong paths from a visual type to target DN types."""

    results = []

    stack = [
        (
            start_type,
            [start_type],
            [],
        )
    ]

    while stack:
        current, path, synapses = stack.pop()

        if len(path) - 1 >= MAX_HOPS:
            continue

        for next_type, connection_synapses in (
            adjacency.get(current, [])
        ):
            if next_type in path:
                continue

            new_path = path + [next_type]
            new_synapses = (
                synapses
                + [connection_synapses]
            )

            if next_type in target_types:
                results.append(
                    {
                        "start": start_type,
                        "target": next_type,
                        "path": new_path,
                        "synapses": new_synapses,
                        "bottleneck": min(
                            new_synapses
                        ),
                    }
                )

            stack.append(
                (
                    next_type,
                    new_path,
                    new_synapses,
                )
            )

    results.sort(
        key=lambda item: (
            item["bottleneck"],
            sum(item["synapses"]),
        ),
        reverse=True,
    )

    return results


def select_neurons(
    graph,
    cell_type: str,
    limit: int = 20,
) -> list[int]:
    """Select representative neurons of a cell type."""

    neurons = []

    for neuron_id, data in graph.nodes(
        data=True
    ):
        if data.get("cell_type") != cell_type:
            continue

        neurons.append(neuron_id)

        if len(neurons) >= limit:
            break

    return neurons


def main() -> None:
    print("Loading connectome...")

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
        "Building cell-type network..."
    )

    adjacency = build_cell_type_network(
        graph
    )

    print(
        f"Cell types: "
        f"{len(adjacency):,}"
    )

    print()
    print(
        "Finding visual → DN paths..."
    )

    all_paths = []

    for visual_type in sorted(
        VISUAL_TYPES
    ):
        paths = find_paths(
            visual_type,
            TARGET_TYPES,
            adjacency,
        )

        all_paths.extend(paths)

        print(
            f"{visual_type}: "
            f"{len(paths)} paths"
        )

    print()
    print(
        "=== Best visual → DN paths ==="
    )

    all_paths.sort(
        key=lambda item: (
            item["bottleneck"],
            sum(item["synapses"]),
        ),
        reverse=True,
    )

    for index, result in enumerate(
        all_paths[:20],
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{result['start']} → "
            f"{result['target']}"
        )

        print(
            f"    path: "
            f"{' → '.join(result['path'])}"
        )

        print(
            f"    synapses: "
            f"{' → '.join(map(str, result['synapses']))}"
        )

        print(
            f"    bottleneck: "
            f"{result['bottleneck']:,}"
        )

    if not all_paths:
        print(
            "No visual → DN paths found."
        )
        return

    print()
    print(
        "=== Neural simulation ==="
    )

    # Use the strongest path as the first
    # concrete simulation experiment.
    best = all_paths[0]

    print(
        f"Selected path: "
        f"{' → '.join(best['path'])}"
    )

    start_type = best["start"]
    target_type = best["target"]

    input_neurons = select_neurons(
        graph,
        start_type,
        limit=20,
    )

    target_neurons = select_neurons(
        graph,
        target_type,
        limit=100,
    )

    print(
        f"Input neurons ({start_type}): "
        f"{len(input_neurons)}"
    )

    print(
        f"Target neurons ({target_type}): "
        f"{len(target_neurons)}"
    )

    if not input_neurons:
        print(
            "No input neurons found."
        )
        return

    if not target_neurons:
        print(
            "No target neurons found."
        )
        return

    # Extract only the local neighborhood
    # around the selected input neurons.
    simulation_nodes = set()

    for neuron_id in input_neurons:
        simulation_nodes.add(
            neuron_id
        )

        neighbors = graph.neighbors(
            neuron_id
        )

        for neighbor in neighbors:
            simulation_nodes.add(
                neighbor
            )

    subgraph = graph.subgraph(
        simulation_nodes
    ).copy()

    simulation = NeuralSimulation(
        subgraph,
        threshold=1.0,
        decay=0.9,
        synapse_scale=10.0,
    )

    simulation.stimulate_many(
        input_neurons,
        signal=1.0,
    )

    history = simulation.run(
        steps=10
    )

    print()
    print(
        "Simulation firing counts:"
    )

    for step, fired in enumerate(
        history
    ):
        print(
            f"Step {step:2}: "
            f"{len(fired):,} neurons fired"
        )

    target_counts = (
        simulation.get_firing_count(
            history,
            target_neurons,
        )
    )

    fired_targets = {
        neuron_id: count
        for neuron_id, count
        in target_counts.items()
        if count > 0
    }

    print()
    print(
        f"Target {target_type} "
        f"firing neurons: "
        f"{len(fired_targets)}"
    )

    if fired_targets:
        print(
            "SUCCESS: signal reached "
            f"{target_type}"
        )
    else:
        print(
            "NO TARGET FIRING: "
            "the current local simulation "
            "did not reach the target."
        )


if __name__ == "__main__":
    main()
    