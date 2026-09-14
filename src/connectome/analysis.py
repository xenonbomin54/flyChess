from __future__ import annotations

from collections import deque

import networkx as nx
import pandas as pd


def count_neurons_by_type(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """Count neurons for each cell type."""

    counts: dict[str, int] = {}

    for _, data in graph.nodes(data=True):
        cell_type = data.get("cell_type", "Unknown")
        counts[cell_type] = counts.get(cell_type, 0) + 1

    result = pd.DataFrame(
        [
            {
                "cell_type": cell_type,
                "neuron_count": count,
            }
            for cell_type, count in counts.items()
        ]
    )

    return result.sort_values(
        "neuron_count",
        ascending=False,
    ).reset_index(drop=True)


def summarize_cell_types(
    graph: nx.DiGraph,
) -> pd.DataFrame:
    """Summarize neuron count and total incoming/outgoing synapses."""

    stats: dict[str, dict[str, int]] = {}

    for _, data in graph.nodes(data=True):
        cell_type = data.get("cell_type", "Unknown")

        if cell_type not in stats:
            stats[cell_type] = {
                "neuron_count": 0,
                "outgoing_synapses": 0,
                "incoming_synapses": 0,
            }

        stats[cell_type]["neuron_count"] += 1

    for source, target, data in graph.edges(data=True):
        source_type = graph.nodes[source].get("cell_type", "Unknown")
        target_type = graph.nodes[target].get("cell_type", "Unknown")

        syn_count = int(data.get("syn_count", 0))

        if source_type not in stats:
            stats[source_type] = {
                "neuron_count": 0,
                "outgoing_synapses": 0,
                "incoming_synapses": 0,
            }

        if target_type not in stats:
            stats[target_type] = {
                "neuron_count": 0,
                "outgoing_synapses": 0,
                "incoming_synapses": 0,
            }

        stats[source_type]["outgoing_synapses"] += syn_count
        stats[target_type]["incoming_synapses"] += syn_count

    result = pd.DataFrame(
        [
            {
                "cell_type": cell_type,
                **values,
            }
            for cell_type, values in stats.items()
        ]
    )

    return result.sort_values(
        "outgoing_synapses",
        ascending=False,
    ).reset_index(drop=True)


def get_strong_connections(
    graph: nx.DiGraph,
    cell_type: str,
    direction: str = "out",
    top_k: int = 20,
) -> pd.DataFrame:
    """Get strongest connections involving a cell type."""

    connections: dict[str, dict[str, int]] = {}

    for source, target, data in graph.edges(data=True):
        source_type = graph.nodes[source].get("cell_type", "Unknown")
        target_type = graph.nodes[target].get("cell_type", "Unknown")

        syn_count = int(data.get("syn_count", 0))

        if direction == "out" and source_type == cell_type:
            key = target_type

        elif direction == "in" and target_type == cell_type:
            key = source_type

        else:
            continue

        if key not in connections:
            connections[key] = {
                "syn_count": 0,
                "connection_count": 0,
            }

        connections[key]["syn_count"] += syn_count
        connections[key]["connection_count"] += 1

    result = pd.DataFrame(
        [
            {
                "cell_type": target_type,
                **values,
            }
            for target_type, values in connections.items()
        ]
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "syn_count",
                "connection_count",
            ]
        )

    return result.sort_values(
        ["syn_count", "connection_count"],
        ascending=False,
    ).head(top_k).reset_index(drop=True)


def get_cell_type_network(
    graph: nx.DiGraph,
    min_synapses: int = 100,
) -> pd.DataFrame:
    """Build an aggregated cell-type connection network.

    All neuron-to-neuron connections are first aggregated by
    pre/post cell type. The minimum synapse threshold is then
    applied to the aggregated cell-type connection.
    """

    connections: dict[tuple[str, str], dict[str, int]] = {}

    for source, target, data in graph.edges(data=True):
        source_type = graph.nodes[source].get("cell_type", "Unknown")
        target_type = graph.nodes[target].get("cell_type", "Unknown")

        if source_type == "Unknown" or target_type == "Unknown":
            continue

        syn_count = int(data.get("syn_count", 0))

        key = (source_type, target_type)

        if key not in connections:
            connections[key] = {
                "syn_count": 0,
                "connection_count": 0,
            }

        connections[key]["syn_count"] += syn_count
        connections[key]["connection_count"] += 1

    result = pd.DataFrame(
        [
            {
                "pre_type": pre_type,
                "post_type": post_type,
                **values,
            }
            for (pre_type, post_type), values in connections.items()
            if values["syn_count"] >= min_synapses
        ]
    )

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


def get_cell_type_neighbors(
    graph: nx.DiGraph,
    cell_type: str,
    direction: str = "out",
    min_synapses: int = 1,
    top_k: int = 20,
) -> list[str]:
    """Get neighboring cell types connected to a given cell type."""

    connections = get_strong_connections(
        graph=graph,
        cell_type=cell_type,
        direction=direction,
        top_k=top_k,
    )

    if connections.empty:
        return []

    if min_synapses > 1:
        connections = connections[
            connections["syn_count"] >= min_synapses
        ]

    return connections["cell_type"].tolist()


def trace_cell_type_path(
    graph: nx.DiGraph,
    start_type: str,
    max_hops: int = 3,
    min_synapses: int = 100,
    top_k_per_hop: int = 10,
) -> pd.DataFrame:
    """Trace a cell-type-level pathway starting from a cell type.

    Cell-type connections are aggregated first. The minimum
    synapse threshold is applied to those aggregated connections.
    """

    network = get_cell_type_network(
        graph,
        min_synapses=min_synapses,
    )

    if network.empty:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "hop",
                "parent_type",
                "syn_count",
            ]
        )

    outgoing: dict[str, list[tuple[str, int]]] = {}

    for row in network.itertuples(index=False):
        outgoing.setdefault(row.pre_type, []).append(
            (
                row.post_type,
                int(row.syn_count),
            )
        )

    for source_type in outgoing:
        outgoing[source_type].sort(
            key=lambda item: item[1],
            reverse=True,
        )

    queue = deque(
        [
            (
                start_type,
                0,
                None,
                0,
            )
        ]
    )

    visited = {start_type}
    results = []

    while queue:
        current_type, hop, parent_type, syn_count = queue.popleft()

        results.append(
            {
                "cell_type": current_type,
                "hop": hop,
                "parent_type": parent_type,
                "syn_count": syn_count,
            }
        )

        if hop >= max_hops:
            continue

        neighbors = outgoing.get(current_type, [])

        for target_type, target_synapses in neighbors[:top_k_per_hop]:
            if target_type in visited:
                continue

            visited.add(target_type)

            queue.append(
                (
                    target_type,
                    hop + 1,
                    current_type,
                    target_synapses,
                )
            )

    return pd.DataFrame(results)


def find_cell_types_by_keywords(
    graph: nx.DiGraph,
    keywords: list[str],
) -> pd.DataFrame:
    """Find cell types whose names contain any of the given keywords."""

    counts = count_neurons_by_type(graph)

    if counts.empty:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "neuron_count",
                "matched_keyword",
            ]
        )

    rows = []

    for row in counts.itertuples(index=False):
        cell_type = str(row.cell_type)

        matches = [
            keyword
            for keyword in keywords
            if keyword.lower() in cell_type.lower()
        ]

        if matches:
            rows.append(
                {
                    "cell_type": cell_type,
                    "neuron_count": int(row.neuron_count),
                    "matched_keyword": ", ".join(matches),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "neuron_count",
                "matched_keyword",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "neuron_count",
        ascending=False,
    ).reset_index(drop=True)


def get_cell_type_summary(
    graph: nx.DiGraph,
    cell_type: str,
) -> pd.DataFrame:
    """Get neuron count and total input/output synapses for a cell type."""

    summary = summarize_cell_types(graph)

    result = summary[
        summary["cell_type"] == cell_type
    ].copy()

    return result.reset_index(drop=True)


def analyze_cell_type(
    graph: nx.DiGraph,
    cell_type: str,
    top_k: int = 15,
    min_synapses: int = 100,
) -> dict[str, pd.DataFrame]:
    """Analyze the main incoming and outgoing connections of a cell type."""

    summary = get_cell_type_summary(
        graph,
        cell_type,
    )

    incoming = get_strong_connections(
        graph=graph,
        cell_type=cell_type,
        direction="in",
        top_k=top_k,
    )

    outgoing = get_strong_connections(
        graph=graph,
        cell_type=cell_type,
        direction="out",
        top_k=top_k,
    )

    if min_synapses > 1:
        incoming = incoming[
            incoming["syn_count"] >= min_synapses
        ].reset_index(drop=True)

        outgoing = outgoing[
            outgoing["syn_count"] >= min_synapses
        ].reset_index(drop=True)

    return {
        "summary": summary,
        "incoming": incoming,
        "outgoing": outgoing,
    }


def trace_specific_path(
    graph: nx.DiGraph,
    start_type: str,
    target_type: str,
    max_hops: int = 5,
    min_synapses: int = 1000,
    top_k_per_node: int = 10,
) -> list[dict]:
    """Find a strong cell-type path between two cell types."""

    network = get_cell_type_network(
        graph,
        min_synapses=min_synapses,
    )

    if network.empty:
        return []

    outgoing: dict[str, list[tuple[str, int]]] = {}

    for row in network.itertuples(index=False):
        outgoing.setdefault(row.pre_type, []).append(
            (
                row.post_type,
                int(row.syn_count),
            )
        )

    for source_type in outgoing:
        outgoing[source_type].sort(
            key=lambda item: item[1],
            reverse=True,
        )

    queue = deque(
        [
            (
                start_type,
                [start_type],
                [],
            )
        ]
    )

    visited = {start_type}

    while queue:
        current_type, path, connections = queue.popleft()

        if current_type == target_type:
            return [
                {
                    "from_type": source,
                    "to_type": target,
                    "syn_count": syn_count,
                }
                for source, target, syn_count in connections
            ]

        if len(path) - 1 >= max_hops:
            continue

        for next_type, syn_count in outgoing.get(
            current_type,
            [],
        )[:top_k_per_node]:

            if next_type in visited:
                continue

            visited.add(next_type)

            queue.append(
                (
                    next_type,
                    path + [next_type],
                    connections + [
                        (
                            current_type,
                            next_type,
                            syn_count,
                        )
                    ],
                )
            )

    return []