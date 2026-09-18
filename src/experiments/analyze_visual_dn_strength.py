from __future__ import annotations

import argparse
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

DN_TYPES = [
    "DNp15",
    "DNb03",
    "DNp26",
    "DNae002",
    "DNae010",
    "DNa02",
    "DNp18",
    "DNa16",
    "DNg41",
    "DNp31",
]


def load_data(
    connections_path: str,
    annotations_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    connections = pd.read_csv(
        connections_path,
        compression="infer",
    )

    annotations = pd.read_csv(
        annotations_path,
        sep="\t",
        low_memory=False,
    )

    print(
        f"Connection rows: {len(connections):,}"
    )
    print(
        f"Annotation rows: {len(annotations):,}"
    )

    return connections, annotations


def normalize_annotations(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    annotations = annotations.copy()

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

    for column in [
        "cell_type",
        "flow",
        "super_class",
    ]:
        annotations[column] = (
            annotations[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return annotations


def select_visual_neurons(
    annotations: pd.DataFrame,
    per_type: int,
) -> dict[str, list[int]]:
    result = {}

    for visual_type in VISUAL_TYPES:
        rows = annotations[
            annotations["cell_type"]
            == visual_type
        ]

        ids = (
            rows["root_id"]
            .drop_duplicates()
            .tolist()
        )

        result[visual_type] = ids[:per_type]

    return result


def select_dn_neurons(
    annotations: pd.DataFrame,
) -> dict[int, str]:
    descending = annotations[
        (
            annotations["flow"].str.lower()
            == "efferent"
        )
        & (
            annotations["super_class"].str.lower()
            == "descending"
        )
        & (
            annotations["cell_type"].isin(DN_TYPES)
        )
    ]

    return dict(
        zip(
            descending["root_id"],
            descending["cell_type"],
        )
    )


def build_type_map(
    annotations: pd.DataFrame,
) -> dict[int, str]:
    rows = annotations[
        [
            "root_id",
            "cell_type",
        ]
    ].drop_duplicates(
        subset=["root_id"]
    )

    return dict(
        zip(
            rows["root_id"],
            rows["cell_type"],
        )
    )


def build_outgoing_totals(
    graph,
) -> dict[int, int]:
    totals = {}

    for source in graph.nodes:
        total = 0

        for target in graph.successors(source):
            total += int(
                graph[source][target].get(
                    "syn_count",
                    0,
                )
            )

        totals[source] = total

    return totals


def normalized_weight(
    graph,
    outgoing_totals: dict[int, int],
    source: int,
    target: int,
) -> float:
    synapses = float(
        graph[source][target].get(
            "syn_count",
            0,
        )
    )

    total = float(
        outgoing_totals.get(
            source,
            0,
        )
    )

    if total <= 0:
        return 0.0

    return synapses / total


def strongest_neighbors(
    graph,
    outgoing_totals: dict[int, int],
    node: int,
    top_k: int,
    min_synapses: int,
) -> list[tuple[int, int, float]]:
    neighbors = []

    for target in graph.successors(node):
        synapses = int(
            graph[node][target].get(
                "syn_count",
                0,
            )
        )

        if synapses < min_synapses:
            continue

        weight = normalized_weight(
            graph,
            outgoing_totals,
            node,
            target,
        )

        neighbors.append(
            (
                target,
                synapses,
                weight,
            )
        )

    neighbors.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return neighbors[:top_k]


def search_paths(
    graph,
    outgoing_totals: dict[int, int],
    start: int,
    dn_roots: dict[int, str],
    max_hops: int,
    top_k: int,
    min_synapses: int,
    max_paths: int,
) -> list[dict]:
    """
    Beam-like DFS using normalized outgoing
    synaptic weight as the primary path score.
    """

    results = []

    stack = [
        (
            start,
            [start],
            [],
            1.0,
        )
    ]

    while stack and len(results) < max_paths:
        (
            node,
            path_nodes,
            edge_weights,
            path_strength,
        ) = stack.pop()

        hops = len(path_nodes) - 1

        if (
            node != start
            and node in dn_roots
        ):
            results.append(
                {
                    "dn_root_id": node,
                    "dn_type": dn_roots[node],
                    "path_nodes": path_nodes,
                    "edge_weights": edge_weights,
                    "path_strength": path_strength,
                    "min_edge_weight": min(
                        edge_weights
                    ),
                    "hops": hops,
                }
            )

            continue

        if hops >= max_hops:
            continue

        neighbors = strongest_neighbors(
            graph,
            outgoing_totals,
            node,
            top_k=top_k,
            min_synapses=min_synapses,
        )

        neighbors.sort(
            key=lambda x: x[2],
            reverse=False,
        )

        for (
            target,
            synapses,
            weight,
        ) in neighbors:
            if target in path_nodes:
                continue

            stack.append(
                (
                    target,
                    path_nodes + [target],
                    edge_weights + [weight],
                    path_strength * weight,
                )
            )

    return results


def path_to_string(
    path_nodes: list[int],
    type_map: dict[int, str],
) -> str:
    return " -> ".join(
        type_map.get(
            node,
            str(node),
        )
        for node in path_nodes
    )


def run_analysis(
    graph,
    annotations: pd.DataFrame,
    per_type: int,
    max_hops: int,
    top_k: int,
    min_synapses: int,
    max_paths_per_neuron: int,
) -> pd.DataFrame:
    visual_neurons = select_visual_neurons(
        annotations,
        per_type,
    )

    dn_roots = select_dn_neurons(
        annotations
    )

    type_map = build_type_map(
        annotations
    )

    outgoing_totals = build_outgoing_totals(
        graph
    )

    all_results = []

    print()
    print("=" * 70)
    print("NORMALIZED VISUAL -> DN ANALYSIS")
    print("=" * 70)

    for visual_type in VISUAL_TYPES:
        roots = visual_neurons[
            visual_type
        ]

        print(
            f"{visual_type}: "
            f"{len(roots)} neurons"
        )

        for index, root in enumerate(
            roots,
            start=1,
        ):
            if root not in graph:
                continue

            paths = search_paths(
                graph=graph,
                outgoing_totals=outgoing_totals,
                start=root,
                dn_roots=dn_roots,
                max_hops=max_hops,
                top_k=top_k,
                min_synapses=min_synapses,
                max_paths=max_paths_per_neuron,
            )

            for result in paths:
                result["visual_type"] = (
                    visual_type
                )

                result["visual_root_id"] = (
                    root
                )

                result["path"] = path_to_string(
                    result["path_nodes"],
                    type_map,
                )

                result["edge_weights"] = (
                    ",".join(
                        f"{weight:.8f}"
                        for weight in result[
                            "edge_weights"
                        ]
                    )
                )

                all_results.append(
                    result
                )

            if index % 10 == 0:
                print(
                    f"  {index}/{len(roots)}"
                )

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(
        all_results
    )


def print_summary(
    results: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("NORMALIZED DN STRENGTH")
    print("=" * 70)

    summary = (
        results
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
            max_strength=(
                "path_strength",
                "max",
            ),
            mean_strength=(
                "path_strength",
                "mean",
            ),
            max_edge=(
                "min_edge_weight",
                "max",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "visual_type",
            "max_strength",
        ],
        ascending=[
            True,
            False,
        ],
    )

    for visual_type in VISUAL_TYPES:
        rows = summary[
            summary["visual_type"]
            == visual_type
        ].head(10)

        print()
        print(
            f"[{visual_type}]"
        )

        for row in rows.itertuples(
            index=False
        ):
            print(
                f"  {row.dn_type:>10} "
                f"paths={row.paths:>3} "
                f"max={row.max_strength:.8e} "
                f"mean={row.mean_strength:.8e}"
            )


def print_cross_visual_summary(
    results: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("CROSS-VISUAL DN COMPARISON")
    print("=" * 70)

    summary = (
        results
        .groupby("dn_type")
        .agg(
            visual_types=(
                "visual_type",
                "nunique",
            ),
            max_strength=(
                "path_strength",
                "max",
            ),
            mean_strength=(
                "path_strength",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "max_strength",
            ascending=False,
        )
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8e}"
            ),
        )
    )


def print_best_paths(
    results: pd.DataFrame,
    limit: int,
) -> None:
    print()
    print("=" * 70)
    print(
        f"TOP {limit} NORMALIZED PATHS"
    )
    print("=" * 70)

    top = (
        results
        .sort_values(
            "path_strength",
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
            f"  strength: "
            f"{row['path_strength']:.8e}"
        )

        print(
            f"  min edge: "
            f"{row['min_edge_weight']:.8e}"
        )

        print(
            f"  hops: "
            f"{row['hops']}"
        )

        print(
            f"  path:"
        )

        print(
            f"    {row['path']}"
        )

        print(
            f"  normalized edges:"
        )

        print(
            f"    {row['edge_weights']}"
        )


def save_results(
    results: pd.DataFrame,
    output_path: str,
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "visual_type",
        "visual_root_id",
        "dn_type",
        "dn_root_id",
        "hops",
        "path_strength",
        "min_edge_weight",
        "path",
        "edge_weights",
    ]

    results[
        columns
    ].to_csv(
        path,
        index=False,
    )

    print()
    print(
        f"Saved: {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--connections",
        default=(
            "data/raw/"
            "connections_princeton.csv.gz"
        ),
    )

    parser.add_argument(
        "--annotations",
        default=(
            "data/raw/"
            "neuron_annotations.tsv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "visual_to_dn_strength.csv"
        ),
    )

    parser.add_argument(
        "--per-type",
        type=int,
        default=30,
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
        "--top-paths",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    connections, annotations = load_data(
        args.connections,
        args.annotations,
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
        f"Nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges():,}"
    )

    results = run_analysis(
        graph=graph,
        annotations=annotations,
        per_type=args.per_type,
        max_hops=args.max_hops,
        top_k=args.top_k,
        min_synapses=args.min_synapses,
        max_paths_per_neuron=(
            args.max_paths_per_neuron
        ),
    )

    if results.empty:
        print()
        print(
            "No paths found."
        )
        return

    print_summary(
        results
    )

    print_cross_visual_summary(
        results
    )

    print_best_paths(
        results,
        args.top_paths,
    )

    save_results(
        results,
        args.output,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()