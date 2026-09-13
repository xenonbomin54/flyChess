import pandas as pd

from src.connectome.graph import build_graph


DATA_PATH = "data/raw/connections_princeton.csv.gz"


def load_connections(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the FlyWire connection table."""
    return pd.read_csv(path)


if __name__ == "__main__":
    print("Loading connectome data...")

    df = load_connections()
    graph = build_graph(df)

    print(f"Connections: {len(df):,}")
    print(f"Graph nodes: {graph.number_of_nodes():,}")
    print(f"Graph edges: {graph.number_of_edges():,}")
    print("Connectome loaded successfully.")