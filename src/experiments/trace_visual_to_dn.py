from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.connectome.graph import build_graph


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


DEFAULT_DN_TYPES = [
    "DNp15",
    "DNb03",
    "DNg41",
    "DNae002",
    "DNbe001",
    "DNp26",
    "DNae003",
    "DNa02",
    "DNge006",
    "DNae010",
    "DNge086",
    "DNg71",
    "DNa16",
    "DNg05_a",
    "DNg110",
    "DNge045",
    "DNpe019",
    "DNp18",
    "DNb01",
    "DNa13",
    "DNp31",
    "DNg46",
    "DNa10",
    "DNge043",
    "DNp22",
    "DNg78",
    "DNg42",
    "DNa03",
    "DNge037",
    "DNp51",
    "DNge026",
    "DNg04",
    "DNa11",
    "DNa09",
]


def load_connectome(
    connections_path: str,
) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("LOADING CONNECTOME")
    print("=" * 70)

    df = pd.read_csv(
        connections_path,
        compression="infer",
    )

    print(
        f"Connection rows: {len(df):,}"
    )

    return df


def load_annotations(
    annotation_path: str,
) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("LOADING ANNOTATIONS")
    print("=" * 70)

    path = Path(annotation_path)

    df = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    print(
        f"Annotation rows: {len(df):,}"
    )

    print(
        f"Annotation columns: {len(df.columns)}"
    )

    return df


def normalize_annotations(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize annotation column names."""

    rename_map = {}

    aliases = {
        "superclass": "super_class",
        "superClass": "super_class",
        "cellClass": "cell_class",
        "cellclass": "cell_class",
        "cellType": "cell_type",
        "celltype": "cell_type",
        "rootId": "root_id",
        "rootID": "root_id",
    }

    for source, target in aliases.items():
        if (
            source in annotations.columns
            and target not in annotations.columns
        ):
            rename_map[source] = target

    if rename_map:
        annotations = annotations.rename(
            columns=rename_map
        )

    required = [
        "root_id",
        "cell_type",
        "flow",
        "super_class",
    ]

    missing = [
        column
        for column in required
        if column not in annotations.columns
    ]

    if missing:
        raise ValueError(
            "Missing required annotation columns: "
            + ", ".join(missing)
        )

    annotations["root_id"] = pd.to_numeric(
        annotations["root_id"],
        errors="coerce",
    )

    annotations = annotations.dropna(
        subset=["root_id"]
    )

    annotations["root_id"] = (
        annotations["root_id"]
        .astype("int64")
    )

    annotations["cell_type"] = (
        annotations["cell_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    annotations["flow"] = (
        annotations["flow"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    annotations["super_class"] = (
        annotations["super_class"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return annotations


def select_visual_neurons(
    annotations: pd.DataFrame,
    per_type: int,
) -> dict[str, list[int]]:
    """Select representative neurons for each visual type."""

    result = {}

    for visual_type in VISUAL_TYPES:
        rows = annotations[
            annotations["cell_type"]
            == visual_type
        ]

        root_ids = (
            rows["root_id"]
            .drop_duplicates()
            .tolist()
        )

        root_ids = root_ids[:per_type]

        result[visual_type] = root_ids

        print(
            f"{visual_type:>4}: "
            f"{len(root_ids):>3} neurons"
        )

    return result


def select_dn_types(
    annotations: pd.DataFrame,
) -> set[str]:
    """Select descending neuron cell types."""

    descending = annotations[
        (
            annotations["flow"]
            .str.lower()
            == "efferent"
        )
        & (
            annotations["super_class"]
            .str.lower()
            == "descending"
        )
    ]

    available_types = set(
        descending["cell_type"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    selected = (
        available_types
        & set(DEFAULT_DN_TYPES)
    )

    print()
    print(
        f"Candidate DN types available: "
        f"{len(selected)}"
    )

    for cell_type in sorted(selected):
        print(
            f"  {cell_type}"
        )

    return selected


def build_neuron_type_map(
    annotations: pd.DataFrame,
) -> dict[int, str]:
    """Map root ID to cell type."""

    subset = annotations[
        [
            "root_id",
            "cell_type",
        ]
    ].drop_duplicates(
        subset=["root_id"]
    )

    return dict(
        zip(
            subset["root_id"],
            subset["cell_type"],
        )
    )


def build_dn_root_map(
    annotations: pd.DataFrame,
    dn_types: set[str],
) -> dict[int, str]:
    """Map DN root IDs to DN cell types."""

    descending = annotations[
        (
            annotations["flow"]
            .str.lower()
            == "efferent"
        )
        & (
            annotations["super_class"]
            .str.lower()
            == "descending"
        )
        & (
            annotations["cell_type"]
            .isin(dn_types)
        )
    ]

    return dict(
        zip(
            descending["root_id"],
            descending["cell_type"],
        )
    )


def rank_neighbors(
    graph,
    node: int,
    top_k: int,
    min_synapses: int,
) -> list[tuple[int, int]]:
    """Return strongest outgoing neighbors."""

    neighbors = []

    for target in graph.successors(node):
        data = graph[node][target]

        syn_count = int(
            data.get(
                "syn_count",
                0,
            )
        )

        if syn_count < min_synapses:
            continue

        neighbors.append(
            (
                target,
                syn_count,
            )
        )

    neighbors.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return neighbors[:top_k]


def find_paths_to_dn(
    graph,
    start_node: int,
    dn_root_map: dict[int, str],
    type_map: dict[int, str],
    max_hops: int,
    top_k: int,
    min_synapses: int,
    max_paths: int,
) -> list[dict]:
    """
    Trace strong outgoing paths from one visual neuron
    to candidate descending neurons.
    """

    paths = []

    stack = [
        (
            start_node,
            [start_node],
            [],
        )
    ]

    while stack and len(paths) < max_paths:
        node, path_nodes, path_weights = stack.pop()

        hops = len(path_nodes) - 1

        if hops > max_hops:
            continue

        if (
            node != start_node
            and node in dn_root_map
        ):
            paths.append(
                {
                    "dn_root_id": node,
                    "dn_type": dn_root_map[node],
                    "path_nodes": path_nodes.copy(),
                    "path_weights": path_weights.copy(),
                    "hops": hops,
                    "total_synapses": sum(
                        path_weights
                    ),
                    "bottleneck": min(
                        path_weights
                    )
                    if path_weights
                    else 0,
                }
            )

            continue

        if hops == max_hops:
            continue

        neighbors = rank_neighbors(
            graph,
            node,
            top_k=top_k,
            min_synapses=min_synapses,
        )

        for target, syn_count in reversed(
            neighbors
        ):
            if target in path_nodes:
                continue

            stack.append(
                (
                    target,
                    path_nodes + [target],
                    path_weights + [syn_count],
                )
            )

    return paths


def format_path(
    path: dict,
    type_map: dict[int, str],
) -> str:
    """Format a neuron path using cell types."""

    labels = []

    for node in path["path_nodes"]:
        labels.append(
            type_map.get(
                node,
                str(node),
            )
        )

    return " -> ".join(labels)


def aggregate_paths(
    all_paths: list[dict],
    type_map: dict[int, str],
) -> pd.DataFrame:
    """Aggregate neuron-level paths by visual type and DN type."""

    rows = []

    for item in all_paths:
        rows.append(
            {
                "visual_type": item[
                    "visual_type"
                ],
                "visual_root_id": item[
                    "visual_root_id"
                ],
                "dn_type": item[
                    "dn_type"
                ],
                "dn_root_id": item[
                    "dn_root_id"
                ],
                "hops": item[
                    "hops"
                ],
                "bottleneck": item[
                    "bottleneck"
                ],
                "total_synapses": item[
                    "total_synapses"
                ],
                "path": format_path(
                    item,
                    type_map,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    paths_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("VISUAL -> DN SUMMARY")
    print("=" * 70)

    if paths_df.empty:
        print(
            "No visual -> DN paths found."
        )
        return

    grouped = (
        paths_df
        .groupby(
            [
                "visual_type",
                "dn_type",
            ]
        )
        .agg(
            paths=(
                "dn_root_id",
                "count",
            ),
            best_bottleneck=(
                "bottleneck",
                "max",
            ),
            best_total=(
                "total_synapses",
                "max",
            ),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        [
            "visual_type",
            "paths",
            "best_bottleneck",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    print(
        grouped.to_string(
            index=False
        )
    )


def print_top_paths(
    paths_df: pd.DataFrame,
    limit: int,
) -> None:
    print()
    print("=" * 70)
    print(
        f"TOP {limit} VISUAL -> DN PATHS"
    )
    print("=" * 70)

    if paths_df.empty:
        return

    top = (
        paths_df
        .sort_values(
            [
                "bottleneck",
                "total_synapses",
            ],
            ascending=False,
        )
        .head(limit)
    )

    for index, row in top.iterrows():
        print()
        print(
            f"[{index}] "
            f"{row['visual_type']} "
            f"-> {row['dn_type']}"
        )

        print(
            f"  hops: "
            f"{row['hops']}"
        )

        print(
            f"  bottleneck: "
            f"{row['bottleneck']}"
        )

        print(
            f"  total synapses: "
            f"{row['total_synapses']}"
        )

        print(
            f"  path:"
        )

        print(
            f"    {row['path']}"
        )


def print_dn_coverage(
    paths_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("DN COVERAGE ACROSS VISUAL INPUTS")
    print("=" * 70)

    if paths_df.empty:
        return

    coverage = (
        paths_df
        .groupby("dn_type")[
            "visual_type"
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    for dn_type, count in coverage.items():
        print(
            f"{dn_type:>10}: "
            f"{count}/8 visual types"
        )


def save_paths(
    paths_df: pd.DataFrame,
    output_path: str,
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths_df.to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved: {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace strong visual T4/T5 "
            "connectome paths to candidate "
            "descending neurons."
        )
    )

    parser.add_argument(
        "--connections",
        type=str,
        default=(
            "data/raw/"
            "connections_princeton.csv.gz"
        ),
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=(
            "data/raw/"
            "neuron_annotations.tsv"
        ),
    )

    parser.add_argument(
        "--per-type",
        type=int,
        default=30,
        help=(
            "Number of visual neurons "
            "per T4/T5 type."
        ),
    )

    parser.add_argument(
        "--max-hops",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help=(
            "Strongest outgoing neighbors "
            "to expand at each node."
        ),
    )

    parser.add_argument(
        "--min-synapses",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-paths-per-neuron",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/processed/"
            "visual_to_dn_paths.csv"
        ),
    )

    parser.add_argument(
        "--top-paths",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    connections = load_connectome(
        args.connections
    )

    annotations = load_annotations(
        args.annotations
    )

    annotations = normalize_annotations(
        annotations
    )

    print()
    print("=" * 70)
    print("BUILDING GRAPH")
    print("=" * 70)

    graph = build_graph(
        connections
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
    print("=" * 70)
    print("SELECTING VISUAL NEURONS")
    print("=" * 70)

    visual_neurons = select_visual_neurons(
        annotations,
        per_type=args.per_type,
    )

    dn_types = select_dn_types(
        annotations
    )

    type_map = build_neuron_type_map(
        annotations
    )

    dn_root_map = build_dn_root_map(
        annotations,
        dn_types,
    )

    print()
    print(
        f"DN root IDs: "
        f"{len(dn_root_map):,}"
    )

    all_paths = []

    print()
    print("=" * 70)
    print("TRACING VISUAL -> DN PATHS")
    print("=" * 70)

    for visual_type in VISUAL_TYPES:
        roots = visual_neurons[
            visual_type
        ]

        print()
        print(
            f"{visual_type}: "
            f"{len(roots)} starting neurons"
        )

        found_for_type = 0

        for index, root_id in enumerate(
            roots,
            start=1,
        ):
            if root_id not in graph:
                continue

            paths = find_paths_to_dn(
                graph=graph,
                start_node=root_id,
                dn_root_map=dn_root_map,
                type_map=type_map,
                max_hops=args.max_hops,
                top_k=args.top_k,
                min_synapses=args.min_synapses,
                max_paths=args.max_paths_per_neuron,
            )

            for path in paths:
                path["visual_type"] = (
                    visual_type
                )

                path["visual_root_id"] = (
                    root_id
                )

                all_paths.append(
                    path
                )

            found_for_type += len(
                paths
            )

            if index % 10 == 0:
                print(
                    f"  processed "
                    f"{index}/{len(roots)} "
                    f"| paths={found_for_type}"
                )

        print(
            f"  {visual_type} complete: "
            f"{found_for_type} paths"
        )

    paths_df = aggregate_paths(
        all_paths,
        type_map,
    )

    print_summary(
        paths_df
    )

    print_dn_coverage(
        paths_df
    )

    print_top_paths(
        paths_df,
        limit=args.top_paths,
    )

    save_paths(
        paths_df,
        args.output,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        "These are connectivity candidates, "
        "not behavior labels."
    )


if __name__ == "__main__":
    main()