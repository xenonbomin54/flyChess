from __future__ import annotations

import pandas as pd


FIRING_PATH = "data/processed/visual_dn_firing.csv"
ANNOTATIONS_PATH = "data/raw/neuron_annotations.tsv"
OUTPUT_PATH = "data/processed/dn_candidate_analysis.csv"

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


def load_firing() -> pd.DataFrame:
    df = pd.read_csv(FIRING_PATH)

    required = {
        "visual_type",
        "dn_type",
        "firing_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing firing columns: {sorted(missing)}"
        )

    return df


def load_annotations() -> pd.DataFrame:
    df = pd.read_csv(
        ANNOTATIONS_PATH,
        sep="\t",
        low_memory=False,
    )

    df["cell_type"] = (
        df["cell_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def build_firing_matrix(
    firing: pd.DataFrame,
) -> pd.DataFrame:
    matrix = (
        firing
        .pivot_table(
            index="dn_type",
            columns="visual_type",
            values="firing_count",
            aggfunc="sum",
            fill_value=0,
        )
    )

    for visual_type in VISUAL_TYPES:
        if visual_type not in matrix.columns:
            matrix[visual_type] = 0

    matrix = matrix[
        VISUAL_TYPES
    ]

    return matrix


def calculate_statistics(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dn_type, row in matrix.iterrows():
        values = row.astype(float)

        total = values.sum()

        active_visuals = int(
            (values > 0).sum()
        )

        max_firing = values.max()

        if total > 0:
            dominant_visual = values.idxmax()

            dominant_share = (
                max_firing / total
            )
        else:
            dominant_visual = ""
            dominant_share = 0.0

        nonzero_values = values[
            values > 0
        ]

        if len(nonzero_values) > 0:
            mean_active = (
                nonzero_values.mean()
            )
        else:
            mean_active = 0.0

        if total > 0:
            sorted_values = values.sort_values(
                ascending=False
            )

            second = (
                sorted_values.iloc[1]
                if len(sorted_values) > 1
                else 0
            )

            dominance_gap = (
                max_firing - second
            )
        else:
            dominance_gap = 0.0

        rows.append(
            {
                "dn_type": dn_type,
                "total_firing": int(total),
                "active_visuals": active_visuals,
                "max_firing": int(max_firing),
                "dominant_visual": dominant_visual,
                "dominant_share": dominant_share,
                "mean_active_firing": mean_active,
                "dominance_gap": dominance_gap,
            }
        )

    return pd.DataFrame(rows)


def extract_annotations(
    annotations: pd.DataFrame,
    dn_types: list[str],
) -> pd.DataFrame:
    columns = [
        "cell_type",
        "flow",
        "super_class",
        "cell_class",
        "cell_sub_class",
        "supertype",
        "hemibrain_type",
        "ito_lee_hemilineage",
        "hartenstein_hemilineage",
        "top_nt",
        "top_nt_conf",
        "known_nt",
        "side",
        "nerve",
    ]

    available = [
        column
        for column in columns
        if column in annotations.columns
    ]

    result = annotations[
        annotations["cell_type"].isin(dn_types)
    ][available].copy()

    result = (
        result
        .groupby("cell_type", as_index=False)
        .agg(
            {
                column: lambda values: (
                    "; ".join(
                        sorted(
                            {
                                str(value)
                                for value in values
                                if (
                                    pd.notna(value)
                                    and str(value).strip()
                                    and str(value) != "nan"
                                )
                            }
                        )
                    )
                )
                for column in available
                if column != "cell_type"
            }
        )
    )

    return result


def calculate_specialization(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a simple visual selectivity score.

    The score is the dominant visual firing divided by
    total firing. It is descriptive, not a biological
    interpretation.
    """

    rows = []

    for dn_type, row in matrix.iterrows():
        values = row.astype(float)

        total = values.sum()

        if total == 0:
            continue

        dominant = values.max()

        selectivity = (
            dominant / total
        )

        rows.append(
            {
                "dn_type": dn_type,
                "visual_selectivity": selectivity,
            }
        )

    return pd.DataFrame(rows)


def print_candidates(
    result: pd.DataFrame,
) -> None:
    active = result[
        result["total_firing"] > 0
    ].copy()

    active = active.sort_values(
        [
            "active_visuals",
            "total_firing",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print()
    print("=" * 80)
    print("ACTIVE DN CANDIDATES")
    print("=" * 80)

    print(
        f"{'DN':12s}"
        f"{'visuals':>8s}"
        f"{'total':>8s}"
        f"{'max':>8s}"
        f"{'dominant':>12s}"
        f"{'share':>9s}"
    )

    for _, row in active.iterrows():
        print(
            f"{row['dn_type']:12s}"
            f"{int(row['active_visuals']):8d}"
            f"{int(row['total_firing']):8d}"
            f"{int(row['max_firing']):8d}"
            f"{row['dominant_visual']:>12s}"
            f"{row['dominant_share']:9.3f}"
        )


def print_specialized_candidates(
    result: pd.DataFrame,
) -> None:
    active = result[
        result["total_firing"] > 0
    ].copy()

    active = active[
        active["active_visuals"] <= 4
    ]

    active = active.sort_values(
        [
            "dominant_share",
            "total_firing",
        ],
        ascending=False,
    )

    print()
    print("=" * 80)
    print("VISUAL-SPECIALIZED CANDIDATES")
    print("=" * 80)

    for _, row in active.head(15).iterrows():
        print(
            f"{row['dn_type']:12s}"
            f" dominant={row['dominant_visual']:4s}"
            f" share={row['dominant_share']:.3f}"
            f" total={int(row['total_firing'])}"
            f" active_visuals={int(row['active_visuals'])}"
        )


def print_annotation_summary(
    result: pd.DataFrame,
) -> None:
    active = result[
        result["total_firing"] > 0
    ].copy()

    active = active.sort_values(
        "total_firing",
        ascending=False,
    )

    print()
    print("=" * 80)
    print("TOP DN ANNOTATIONS")
    print("=" * 80)

    columns = [
        "dn_type",
        "total_firing",
        "active_visuals",
        "dominant_visual",
        "flow",
        "super_class",
        "hemibrain_type",
        "top_nt",
        "side",
        "nerve",
    ]

    available = [
        column
        for column in columns
        if column in active.columns
    ]

    print(
        active[
            available
        ].head(20).to_string(
            index=False
        )
    )


def main() -> None:
    print("=" * 80)
    print("LOADING DN FIRING DATA")
    print("=" * 80)

    firing = load_firing()

    print(
        f"Firing rows: "
        f"{len(firing):,}"
    )

    print()
    print("=" * 80)
    print("LOADING ANNOTATIONS")
    print("=" * 80)

    annotations = load_annotations()

    print(
        f"Annotation rows: "
        f"{len(annotations):,}"
    )

    print()
    print("=" * 80)
    print("BUILDING FIRING MATRIX")
    print("=" * 80)

    matrix = build_firing_matrix(
        firing
    )

    statistics = calculate_statistics(
        matrix
    )

    specialization = calculate_specialization(
        matrix
    )

    dn_types = matrix.index.tolist()

    annotation_summary = extract_annotations(
        annotations,
        dn_types,
    )

    result = (
        statistics
        .merge(
            specialization,
            on="dn_type",
            how="left",
        )
        .merge(
            annotation_summary,
            left_on="dn_type",
            right_on="cell_type",
            how="left",
        )
    )

    result = result.drop(
        columns=[
            "cell_type"
        ],
        errors="ignore",
    )

    result = result.fillna("")

    output = []

    for _, row in result.iterrows():
        dn_type = row["dn_type"]

        matrix_row = matrix.loc[
            dn_type
        ]

        item = row.to_dict()

        for visual_type in VISUAL_TYPES:
            item[
                f"firing_{visual_type}"
            ] = int(
                matrix_row[
                    visual_type
                ]
            )

        output.append(item)

    result = pd.DataFrame(output)

    result = result.sort_values(
        [
            "total_firing",
            "dominant_share",
        ],
        ascending=False,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_candidates(
        result
    )

    print_specialized_candidates(
        result
    )

    print_annotation_summary(
        result
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()