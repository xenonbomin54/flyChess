from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DN_CANDIDATES = [
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


ANNOTATION_COLUMNS = [
    "root_id",
    "label",
    "side",
    "neuromere",
    "nt_type",
    "input_neuropils",
    "output_neuropils",
    "input_hemisphere",
    "output_hemisphere",
    "flow",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "resolved_type",
    "gene",
    "dimorphism",
    "hemilineage",
    "nerve",
    "nt_type_verified",
    "neuropeptide_verified",
    "body_part",
    "function",
    "name",
    "group",
    "connectivity_tag",
    "mirror_twin_root_id",
    "marker",
    "sensory_in",
    "effector_out",
]


def find_annotation_file(
    explicit_path: str | None,
) -> Path:
    """Find the Codex annotation CSV."""

    if explicit_path:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {path}"
            )

        return path

    candidates = [
        Path("data/raw/annotations.csv"),
        Path("data/raw/codex_annotations.csv"),
        Path("data/raw/cell_annotations.csv"),
        Path("data/raw/consolidated_annotations.csv"),
        Path("data/raw/consolidated_cell_types.csv.gz"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No annotation CSV found.\n"
        "Expected one of:\n"
        + "\n".join(
            f"  - {path}"
            for path in candidates
        )
        + "\n\n"
        "Download the Codex annotation/cell data CSV "
        "and place it under data/raw/."
    )


def load_annotations(
    path: Path,
) -> pd.DataFrame:
    """Load annotation CSV."""

    df = pd.read_csv(path)

    print()
    print("=" * 70)
    print("ANNOTATION DATA")
    print("=" * 70)
    print(f"File: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Available columns:")
    print(", ".join(df.columns))

    return df


def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common Codex column aliases."""

    rename_map = {}

    aliases = {
        "subclass": "sub_class",
        "subClass": "sub_class",
        "superclass": "super_class",
        "superClass": "super_class",
        "resolvedType": "resolved_type",
        "cellType": "cell_type",
        "rootId": "root_id",
        "rootID": "root_id",
        "sensoryIn": "sensory_in",
        "effectorOut": "effector_out",
        "inputNeuropils": "input_neuropils",
        "outputNeuropils": "output_neuropils",
        "inputHemisphere": "input_hemisphere",
        "outputHemisphere": "output_hemisphere",
        "ntType": "nt_type",
        "ntTypeVerified": "nt_type_verified",
    }

    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            rename_map[source] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def filter_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Find candidate DN annotations."""

    if "cell_type" not in df.columns:
        raise ValueError(
            "Annotation file does not contain "
            "'cell_type'."
        )

    result = df[
        df["cell_type"].astype(str).isin(
            DN_CANDIDATES
        )
    ].copy()

    return result


def print_missing_columns(
    df: pd.DataFrame,
) -> None:
    """Print which useful Codex fields are available."""

    print()
    print("=" * 70)
    print("ANNOTATION COVERAGE")
    print("=" * 70)

    for column in ANNOTATION_COLUMNS:
        status = "YES" if column in df.columns else "NO"
        print(f"{status:>3}  {column}")


def print_candidate_summary(
    candidates: pd.DataFrame,
) -> None:
    """Print annotation summary for candidate DNs."""

    print()
    print("=" * 70)
    print("DN CANDIDATE ANNOTATIONS")
    print("=" * 70)

    if candidates.empty:
        print("No candidate DN types found.")
        return

    preferred = [
        "cell_type",
        "resolved_type",
        "flow",
        "super_class",
        "class",
        "sub_class",
        "nerve",
        "function",
        "sensory_in",
        "effector_out",
        "input_neuropils",
        "output_neuropils",
        "side",
        "hemilineage",
        "nt_type",
    ]

    available = [
        column
        for column in preferred
        if column in candidates.columns
    ]

    summary = candidates[
        available
    ].drop_duplicates()

    print(
        summary.to_string(
            index=False
        )
    )


def print_function_groups(
    candidates: pd.DataFrame,
) -> None:
    """Group candidate neurons by functional annotation."""

    print()
    print("=" * 70)
    print("FUNCTION / EFFECTOR SUMMARY")
    print("=" * 70)

    if candidates.empty:
        return

    for column in [
        "flow",
        "super_class",
        "class",
        "sub_class",
        "nerve",
        "function",
        "sensory_in",
        "effector_out",
    ]:
        if column not in candidates.columns:
            continue

        print()
        print(f"[{column}]")

        values = (
            candidates[column]
            .fillna("")
            .astype(str)
            .replace("", "Unknown")
            .value_counts()
        )

        if values.empty:
            print("  Unknown")
            continue

        for value, count in values.items():
            print(
                f"  {value}: {count}"
            )


def print_candidate_table(
    candidates: pd.DataFrame,
) -> None:
    """Print compact one-row-per-cell-type table."""

    print()
    print("=" * 70)
    print("COMPACT DN TABLE")
    print("=" * 70)

    if candidates.empty:
        return

    columns = [
        "cell_type",
        "resolved_type",
        "flow",
        "super_class",
        "class",
        "sub_class",
        "nerve",
        "function",
        "sensory_in",
        "effector_out",
    ]

    available = [
        column
        for column in columns
        if column in candidates.columns
    ]

    sort_columns = [
        column
        for column in [
            "cell_type",
            "resolved_type",
        ]
        if column in available
    ]

    table = candidates[
        available
    ].drop_duplicates()

    if sort_columns:
        table = table.sort_values(
            by=sort_columns
        )

    print(
        table.to_string(
            index=False
        )
    )


def save_result(
    candidates: pd.DataFrame,
    output_path: str,
) -> None:
    """Save candidate annotations."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
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
            "Analyze Codex annotations for "
            "DN candidates from the visual "
            "connectome simulation."
        )
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help=(
            "Path to Codex annotation CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/processed/"
            "dn_candidate_annotations.csv"
        ),
        help=(
            "Output CSV path."
        ),
    )

    args = parser.parse_args()

    annotation_path = find_annotation_file(
        args.annotations
    )

    annotations = load_annotations(
        annotation_path
    )

    annotations = normalize_columns(
        annotations
    )

    print_missing_columns(
        annotations
    )

    candidates = filter_candidates(
        annotations
    )

    print()
    print(
        f"Candidate DN rows: "
        f"{len(candidates):,}"
    )

    print_candidate_summary(
        candidates
    )

    print_function_groups(
        candidates
    )

    print_candidate_table(
        candidates
    )

    save_result(
        candidates,
        args.output,
    )

    print()
    print("=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print(
        "Use flow / class / function / "
        "effector_out annotations to determine "
        "which DN candidates have plausible "
        "motor-output relevance."
    )
    print(
        "Do not assign JUMP / WAIT labels yet."
    )


if __name__ == "__main__":
    main()