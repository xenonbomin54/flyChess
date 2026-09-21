from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/visual_dn_firing.csv"
)

OUTPUT_PATH = Path(
    "data/processed/brain_output.csv"
)


def load_firing_data() -> pd.DataFrame:
    """Load visual input -> DN firing results."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "visual_type",
        "dn_type",
        "firing_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    return df


def analyze_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert DN firing activity into one output per visual input."""

    rows = []

    for visual_type, group in df.groupby(
        "visual_type"
    ):
        group = group.sort_values(
            "firing_count",
            ascending=False,
        )

        total_activity = int(
            group["firing_count"].sum()
        )

        active_neurons = int(
            (group["firing_count"] > 0).sum()
        )

        strongest = group.iloc[0]

        strongest_dn = str(
            strongest["dn_type"]
        )

        strongest_activity = int(
            strongest["firing_count"]
        )

        if total_activity > 0:
            strongest_share = (
                strongest_activity
                / total_activity
            )
        else:
            strongest_share = 0.0

        rows.append(
            {
                "visual_type": visual_type,
                "strongest_dn": strongest_dn,
                "strongest_activity": (
                    strongest_activity
                ),
                "total_activity": (
                    total_activity
                ),
                "active_dns": (
                    active_neurons
                ),
                "strongest_share": (
                    strongest_share
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "visual_type"
    )


def print_results(
    result: pd.DataFrame,
) -> None:
    """Print the summarized neural outputs."""

    print()
    print("=" * 70)
    print("BRAIN OUTPUT")
    print("=" * 70)

    for row in result.itertuples(
        index=False
    ):
        print(
            f"{row.visual_type:>4} "
            f"-> {row.strongest_dn:<10} "
            f"activity={row.strongest_activity:<4} "
            f"total={row.total_activity:<5} "
            f"active={row.active_dns:<3} "
            f"share={row.strongest_share:.3f}"
        )


def main() -> None:
    print("=" * 70)
    print("LOADING NEURAL OUTPUT")
    print("=" * 70)

    df = load_firing_data()

    print(
        f"Firing rows: {len(df):,}"
    )

    result = analyze_output(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_results(result)

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()