from __future__ import annotations

import pandas as pd


def load_connectome(path: str) -> pd.DataFrame:
    """Load connectome connection data."""
    return pd.read_csv(path)


def load_cell_types(path: str) -> pd.DataFrame:
    """Load consolidated cell type data."""
    return pd.read_csv(path)