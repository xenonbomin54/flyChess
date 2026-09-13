import pandas as pd
import networkx as nx


DATA_PATH = "data/raw/connections_princeton.csv.gz"


def load_connections(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the FlyWire connection table."""
    return pd.read_csv(path)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed connectome graph from connection data."""
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


if __name__ == "__main__":
    print("Loading connectome data...")

    df = load_connections()

    print(f"Connections: {len(df):,}")
    print(
        f"Neurons: "
        f"{len(set(df.pre_root_id) | set(df.post_root_id)):,}"
    )

    graph = build_graph(df)

    print(f"Graph nodes: {graph.number_of_nodes():,}")
    print(f"Graph edges: {graph.number_of_edges():,}")
    print("Connectome loaded successfully.")