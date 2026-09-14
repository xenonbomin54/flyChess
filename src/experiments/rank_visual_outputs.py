from __future__ import annotations

from collections import defaultdict

from src.connectome.analysis import get_cell_type_network
from src.connectome.graph import attach_cell_types, build_graph
from src.connectome.loader import load_cell_types, load_connectome


CONNECTIONS_PATH = "data/raw/connections_princeton.csv.gz"
CELL_TYPES_PATH = "data/raw/consolidated_cell_types.csv.gz"

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

MAX_HOPS = 5
MIN_SYNAPSES = 100
TOP_K_PER_NODE = 30
TOP_PATHS_PER_VISUAL = 30


def build_adjacency(network):
    """Build a cell-type adjacency list."""

    adjacency = defaultdict(list)

    for row in network.itertuples(index=False):
        adjacency[row.pre_type].append(
            (
                row.post_type,
                int(row.syn_count),
            )
        )

    for source_type in adjacency:
        adjacency[source_type].sort(
            key=lambda item: item[1],
            reverse=True,
        )

    return adjacency


def is_descending(cell_type: str) -> bool:
    """Return whether a cell type looks like a descending neuron."""

    return (
        cell_type.startswith("DN")
        or cell_type.startswith("DNg")
    )


def find_paths(
    start_type: str,
    adjacency,
) -> list[dict]:
    """Find bounded strong paths from a visual type to DN types."""

    results = []

    stack = [
        (
            start_type,
            [start_type],
            [],
        )
    ]

    while stack:
        current_type, path, synapses = stack.pop()

        hop_count = len(path) - 1

        if hop_count >= MAX_HOPS:
            continue

        neighbors = adjacency.get(
            current_type,
            [],
        )[:TOP_K_PER_NODE]

        for next_type, connection_synapses in neighbors:
            if next_type in path:
                continue

            new_path = path + [next_type]
            new_synapses = synapses + [connection_synapses]

            if is_descending(next_type):
                results.append(
                    {
                        "start_type": start_type,
                        "target_type": next_type,
                        "path": new_path,
                        "synapses": new_synapses,
                        "bottleneck": min(new_synapses),
                        "total_synapses": sum(new_synapses),
                        "hops": len(new_path) - 1,
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
            item["total_synapses"],
        ),
        reverse=True,
    )

    return results[:TOP_PATHS_PER_VISUAL]


def rank_candidates(
    all_paths: list[dict],
):
    """Aggregate path evidence by descending cell type."""

    candidates = defaultdict(
        lambda: {
            "visual_types": set(),
            "path_count": 0,
            "best_bottleneck": 0,
            "best_total_synapses": 0,
            "best_path": [],
        }
    )

    for result in all_paths:
        target = result["target_type"]

        item = candidates[target]

        item["visual_types"].add(
            result["start_type"]
        )

        item["path_count"] += 1

        if (
            result["bottleneck"],
            result["total_synapses"],
        ) > (
            item["best_bottleneck"],
            item["best_total_synapses"],
        ):
            item["best_bottleneck"] = result["bottleneck"]
            item["best_total_synapses"] = result[
                "total_synapses"
            ]
            item["best_path"] = result["path"]

    rows = []

    for target, item in candidates.items():
        rows.append(
            {
                "descending_type": target,
                "visual_type_count": len(
                    item["visual_types"]
                ),
                "visual_types": ", ".join(
                    sorted(item["visual_types"])
                ),
                "path_count": item["path_count"],
                "best_bottleneck": item[
                    "best_bottleneck"
                ],
                "best_total_synapses": item[
                    "best_total_synapses"
                ],
                "best_path": " → ".join(
                    item["best_path"]
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["visual_type_count"],
            row["best_bottleneck"],
            row["path_count"],
            row["best_total_synapses"],
        ),
        reverse=True,
    )

    return rows


def main() -> None:
    print("Loading connectome...")

    connections = load_connectome(
        CONNECTIONS_PATH
    )

    cell_types = load_cell_types(
        CELL_TYPES_PATH
    )

    print(
        f"Connections: {len(connections):,}"
    )

    graph = build_graph(connections)

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
    print("Building cell-type network...")

    network = get_cell_type_network(
        graph,
        min_synapses=MIN_SYNAPSES,
    )

    print(
        f"Cell-type connections: "
        f"{len(network):,}"
    )

    adjacency = build_adjacency(
        network
    )

    print()
    print(
        "Searching visual → descending paths..."
    )

    all_paths = []

    for visual_type in VISUAL_TYPES:
        print(
            f"  Searching {visual_type}..."
        )

        paths = find_paths(
            visual_type,
            adjacency,
        )

        all_paths.extend(paths)

        print(
            f"    Found {len(paths)} "
            f"strong paths"
        )

    candidates = rank_candidates(
        all_paths
    )

    print()
    print(
        "=== Visual → Descending "
        "candidates ==="
    )
    print()

    if not candidates:
        print("No candidates found.")
        return

    for index, row in enumerate(
        candidates[:30],
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{row['descending_type']:<12} "
            f"visual="
            f"{row['visual_type_count']}/8  "
            f"paths="
            f"{row['path_count']:<4} "
            f"bottleneck="
            f"{row['best_bottleneck']:,}"
        )

        print(
            f"    visual types: "
            f"{row['visual_types']}"
        )

        print(
            f"    path: "
            f"{row['best_path']}"
        )

        print()


if __name__ == "__main__":
    main()