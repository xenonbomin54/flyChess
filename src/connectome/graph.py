from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from FlyWire connectome data."""
    graph = nx.DiGraph()

    for row in df.itertuples(index=False):
        graph.add_edge(
            row.pre_root_id,
            row.post_root_id,
            neuropil=row.neuropil,
            syn_count=row.syn_count,
            nt_type=row.nt_type,
        )

    return graph


def get_neurons(
    graph: nx.DiGraph,
    root_id: int,
    direction: str = "both",
) -> list[int]:
    """Get neurons directly connected to a neuron."""
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
    """Get detailed direct connections for a neuron."""
    connections = []

    if direction in ("pre", "both"):
        for source, target, data in graph.in_edges(
            root_id,
            data=True,
        ):
            connections.append(
                {
                    "pre_root_id": source,
                    "post_root_id": target,
                    **data,
                }
            )

    if direction in ("post", "both"):
        for source, target, data in graph.out_edges(
            root_id,
            data=True,
        ):
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
    """Extract a local neural circuit around a neuron."""
    nodes = nx.single_source_shortest_path_length(
        graph.to_undirected(),
        root_id,
        cutoff=hops,
    )

    return graph.subgraph(nodes.keys()).copy()