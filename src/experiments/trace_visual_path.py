from __future__ import annotations

from dataclasses import dataclass

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
    "T4a",
    "TmY16",
    "Mi4",
    "TmY3",
    "LC4",
    "DNp04",
]

TOP_START_NEURONS = 100
TOP_NEIGHBORS = 20
TOP_PATHS = 20


@dataclass
class PathState:
    """A candidate neuron-level path."""

    neurons: list[int]
    synapses: list[int]

    @property
    def bottleneck(self) -> int:
        return min(self.synapses)

    @property
    def total_synapses(self) -> int:
        return sum(self.synapses)


def get_neurons_by_type(
    graph,
    cell_type: str,
) -> list[int]:
    """Return all neurons belonging to a cell type."""

    return [
        neuron_id
        for neuron_id, data in graph.nodes(
            data=True
        )
        if data.get("cell_type") == cell_type
    ]


def get_target_neighbors(
    graph,
    neuron_id: int,
    target_type: str,
) -> list[tuple[int, int]]:
    """Get outgoing neighbors matching a target cell type."""

    results = []

    if neuron_id not in graph:
        return results

    for target in graph.successors(
        neuron_id
    ):
        target_data = graph.nodes[
            target
        ]

        if (
            target_data.get("cell_type")
            != target_type
        ):
            continue

        edge_data = graph[
            neuron_id
        ][target]

        syn_count = int(
            edge_data.get(
                "syn_count",
                0,
            )
        )

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


def find_paths(
    graph,
) -> list[PathState]:
    """Find strong neuron-level paths following PATH."""

    start_type = PATH[0]
    first_neurons = get_neurons_by_type(
        graph,
        start_type,
    )

    print(
        f"{start_type} neurons: "
        f"{len(first_neurons):,}"
    )

    states = []

    # Start from T4a neurons that actually
    # connect to the next cell type.
    next_type = PATH[1]

    for neuron_id in first_neurons:
        neighbors = get_target_neighbors(
            graph,
            neuron_id,
            next_type,
        )

        for target, syn_count in neighbors:
            states.append(
                PathState(
                    neurons=[
                        neuron_id,
                        target,
                    ],
                    synapses=[
                        syn_count,
                    ],
                )
            )

    states.sort(
        key=lambda state: (
            state.bottleneck,
            state.total_synapses,
        ),
        reverse=True,
    )

    states = states[
        :TOP_START_NEURONS
    ]

    print(
        f"Initial T4a → {next_type} "
        f"connections retained: "
        f"{len(states)}"
    )

    # Extend the actual neuron paths one
    # cell-type layer at a time.
    for index in range(1, len(PATH) - 1):
        current_type = PATH[index]
        target_type = PATH[index + 1]

        print(
            f"Tracing "
            f"{current_type} → "
            f"{target_type}..."
        )

        expanded = []

        for state in states:
            current_neuron = (
                state.neurons[-1]
            )

            neighbors = get_target_neighbors(
                graph,
                current_neuron,
                target_type,
            )

            for target, syn_count in neighbors:
                if target in state.neurons:
                    continue

                expanded.append(
                    PathState(
                        neurons=(
                            state.neurons
                            + [target]
                        ),
                        synapses=(
                            state.synapses
                            + [syn_count]
                        ),
                    )
                )

        expanded.sort(
            key=lambda state: (
                state.bottleneck,
                state.total_synapses,
            ),
            reverse=True,
        )

        states = expanded[:TOP_PATHS]

        print(
            f"  candidates: "
            f"{len(states)}"
        )

        if not states:
            print(
                "  No actual neuron-level "
                "connection found."
            )
            return []

    return states


def print_paths(
    graph,
    paths: list[PathState],
) -> None:
    """Print neuron-level paths."""

    print()
    print(
        "=== Actual neuron-level paths ==="
    )
    print()

    if not paths:
        print(
            "No complete neuron-level path found."
        )
        return

    for index, state in enumerate(
        paths,
        start=1,
    ):
        print(
            f"{index:2}. "
            f"bottleneck="
            f"{state.bottleneck:,} "
            f"total="
            f"{state.total_synapses:,}"
        )

        print(
            "    neurons:"
        )

        for neuron_id, cell_type in zip(
            state.neurons,
            PATH,
        ):
            print(
                f"      {cell_type:<8} "
                f"{neuron_id}"
            )

        print(
            "    synapses: "
            + " → ".join(
                map(
                    str,
                    state.synapses,
                )
            )
        )

        print()


def main() -> None:
    print("Loading connectome...")

    connections = load_connectome(
        CONNECTIONS_PATH
    )

    cell_types = load_cell_types(
        CELL_TYPES_PATH
    )

    print(
        f"Connection rows: "
        f"{len(connections):,}"
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
        "Target cell-type path:"
    )
    print(
        "  "
        + " → ".join(PATH)
    )

    print()

    paths = find_paths(
        graph
    )

    print_paths(
        graph,
        paths,
    )

    if paths:
        best = paths[0]

        print(
            "=== VERIFICATION ==="
        )
        print()

        print(
            "Complete neuron-level path found."
        )

        print(
            "Cell types:"
        )

        print(
            "  "
            + " → ".join(PATH)
        )

        print()

        print(
            "Synapses:"
        )

        print(
            "  "
            + " → ".join(
                map(
                    str,
                    best.synapses,
                )
            )
        )

        print()

        print(
            f"Strongest bottleneck: "
            f"{best.bottleneck:,}"
        )

        print()
        print(
            "This confirms that the selected "
            "cell-type pathway contains an "
            "actual neuron-to-neuron path."
        )


if __name__ == "__main__":
    main()