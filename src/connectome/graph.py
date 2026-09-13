from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed connectome graph.

    Multiple rows connecting the same neuron pair are aggregated.
    Synapse counts are summed and neurotransmitter types are preserved.
    """
    graph = nx.DiGraph()

    grouped = (
        df.groupby(["pre_root_id", "post_root_id"], as_index=False)
        .agg(
            syn_count=("syn_count", "sum"),
            nt_type=("nt_type", lambda values: ",".join(sorted(set(values)))),
        )
    )

    for row in grouped.itertuples(index=False):
        graph.add_edge(
            row.pre_root_id,
            row.post_root_id,
            syn_count=int(row.syn_count),
            nt_type=row.nt_type,
        )

    return graph


def attach_cell_types(
    graph: nx.DiGraph,
    cell_types: pd.DataFrame,
) -> nx.DiGraph:
    """Attach cell type information to graph nodes."""
    cell_type_map = (
        cell_types
        .set_index("root_id")["primary_type"]
        .to_dict()
    )

    for neuron_id in graph.nodes:
        graph.nodes[neuron_id]["cell_type"] = (
            cell_type_map.get(neuron_id, "Unknown")
        )

    return graph


def get_neurons(
    graph: nx.DiGraph,
    root_id: int,
    direction: str = "both",
) -> list[int]:
    """Get neurons connected to a neuron."""
    if root_id not in graph:
        return []

    if direction == "pre":
        return list(graph.predecessors(root_id))

    if direction == "post":
        return list(graph.successors(root_id))

    if direction == "both":
        return list(
            set(graph.predecessors(root_id))
            | set(graph.successors(root_id))
        )

    raise ValueError("direction must be 'pre', 'post', or 'both'")


def get_connections(
    graph: nx.DiGraph,
    root_id: int,
    direction: str = "both",
) -> pd.DataFrame:
    """Return connections involving a neuron."""
    connections = []

    if direction in ("pre", "both"):
        for source, target, data in graph.in_edges(root_id, data=True):
            connections.append(
                {
                    "pre_root_id": source,
                    "post_root_id": target,
                    **data,
                }
            )

    if direction in ("post", "both"):
        for source, target, data in graph.out_edges(root_id, data=True):
            connections.append(
                {
                    "pre_root_id": source,
                    "post_root_id": target,
                    **data,
                }
            )

    return pd.DataFrame(connections)


def get_subgraph(
    graph: nx.DiGraph,
    root_id: int,
    hops: int = 1,
) -> nx.DiGraph:
    """Return a subgraph within a given number of hops."""
    nodes = nx.single_source_shortest_path_length(
        graph.to_undirected(),
        root_id,
        cutoff=hops,
    )

    return graph.subgraph(nodes.keys()).copy()


def get_neurons_by_type(
    graph: nx.DiGraph,
    cell_type: str,
) -> list[int]:
    """Get neuron IDs matching a cell type."""
    return [
        neuron_id
        for neuron_id, data in graph.nodes(data=True)
        if data.get("cell_type") == cell_type
    ]


def get_cell_type_connections(
    graph: nx.DiGraph,
    min_synapses: int = 1,
) -> pd.DataFrame:
    """Aggregate connections between cell types.

    Returns one row per cell-type pair with the total synapse count
    and number of neuron-to-neuron connections.
    """
    connections = {}

    for source, target, data in graph.edges(data=True):
        source_type = graph.nodes[source].get("cell_type", "Unknown")
        target_type = graph.nodes[target].get("cell_type", "Unknown")

        if source_type == "Unknown" or target_type == "Unknown":
            continue

        syn_count = int(data.get("syn_count", 0))

        if syn_count < min_synapses:
            continue

        key = (source_type, target_type)

        if key not in connections:
            connections[key] = {
                "pre_type": source_type,
                "post_type": target_type,
                "syn_count": 0,
                "connection_count": 0,
            }

        connections[key]["syn_count"] += syn_count
        connections[key]["connection_count"] += 1

    result = pd.DataFrame(connections.values())

    if result.empty:
        return pd.DataFrame(
            columns=[
                "pre_type",
                "post_type",
                "syn_count",
                "connection_count",
            ]
        )

    return result.sort_values(
        ["syn_count", "connection_count"],
        ascending=False,
    ).reset_index(drop=True)