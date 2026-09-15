from __future__ import annotations

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

PATH = [
    ("T4a", 720575940632444431),
    ("TmY16", 720575940613187481),
    ("Mi4", 720575940620265432),
    ("TmY3", 720575940624591368),
    ("LC4", 720575940625906702),
    ("DNp04", 720575940629757036),
]


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
        "=== Selected neuron path ==="
    )

    for index, (cell_type, neuron_id) in enumerate(
        PATH,
        start=1,
    ):
        actual_type = graph.nodes[
            neuron_id
        ].get("cell_type")

        print(
            f"{index}. "
            f"{cell_type:<8} "
            f"{neuron_id} "
            f"(graph type: {actual_type})"
        )

    print()
    print(
        "Checking actual connections..."
    )

    for index in range(len(PATH) - 1):
        source_type, source = PATH[index]
        target_type, target = PATH[index + 1]

        if not graph.has_edge(
            source,
            target,
        ):
            print(
                f"FAIL: "
                f"{source_type} → {target_type}"
            )
            return

        data = graph[
            source
        ][target]

        syn_count = int(
            data.get("syn_count", 0)
        )

        print(
            f"{source_type:<8} → "
            f"{target_type:<8} "
            f"{syn_count:,} synapses"
        )

    print()
    print(
        "All five neuron-level connections "
        "confirmed."
    )

    # Build ONLY the selected neuron path.
    path_nodes = [
        neuron_id
        for _, neuron_id in PATH
    ]

    path_graph = graph.subgraph(
        path_nodes
    ).copy()

    print()
    print(
        "Path graph:"
    )

    print(
        f"  Nodes: "
        f"{path_graph.number_of_nodes()}"
    )

    print(
        f"  Edges: "
        f"{path_graph.number_of_edges()}"
    )

    print()
    print(
        "=== Single path simulation ==="
    )

    simulation = NeuralSimulation(
        path_graph,
        threshold=1.0,
        decay=0.9,
        synapse_scale=10.0,
    )

    input_type, input_neuron = PATH[0]
    target_type, target_neuron = PATH[-1]

    print()
    print(
        f"Stimulating "
        f"{input_type} "
        f"{input_neuron}"
    )

    simulation.stimulate(
        input_neuron,
        signal=1.0,
    )

    print()

    fired_target = False

    for step in range(10):
        fired = simulation.step()

        fired_types = []

        for neuron_id in fired:
            for cell_type, path_neuron in PATH:
                if neuron_id == path_neuron:
                    fired_types.append(
                        cell_type
                    )

        print(
            f"Step {step:2}: "
            f"{fired_types if fired_types else 'none'}"
        )

        if target_neuron in fired:
            fired_target = True

    print()
    print(
        "=== RESULT ==="
    )

    if fired_target:
        print(
            f"SUCCESS: "
            f"{target_type} fired."
        )
    else:
        print(
            f"FAILED: "
            f"{target_type} did not fire."
        )

    print()
    print(
        "Final neuron activities:"
    )

    for cell_type, neuron_id in PATH:
        activity = simulation.get_activity(
            neuron_id
        )

        print(
            f"  {cell_type:<8} "
            f"{activity:.6f}"
        )


if __name__ == "__main__":
    main()