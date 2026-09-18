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
    "flow",
    "super_class",
    "cell_class",
    "cell_type",
    "side",
    "nerve",
    "hemilineage",
    "nt_type",
    "hemibrain_type",
    "morphology_group",
]


def find_annotation_file(
    explicit_path: str | None,
) -> Path:
    """Find the detailed FlyWire neuron annotation file."""

    if explicit_path:
        path = Path(explicit_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {path}"
            )

        return path

    candidates = [
        Path("data/raw/neuron_annotations.tsv"),
        Path(
            "data/raw/"
            "Supplemental_file1_neuron_annotations.tsv"
        ),
        Path("data/raw/annotations.tsv"),
        Path("data/raw/codex_annotations.tsv"),
        Path("data/raw/cell_annotations.tsv"),
        Path("data/raw/consolidated_annotations.tsv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No detailed neuron annotation file found.\n\n"
        "Expected one of:\n"
        + "\n".join(
            f"  - {path}"
            for path in candidates
        )
        + "\n\n"
        "Download Supplemental_file1_neuron_annotations.tsv "
        "into data/raw/."
    )


def load_annotations(
    path: Path,
) -> pd.DataFrame:
    """Load the FlyWire neuron annotation table."""

    suffix = path.suffix.lower()

    if suffix == ".tsv":
        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )
    else:
        df = pd.read_csv(
            path,
            low_memory=False,
        )

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
    """Normalize annotation column names."""

    rename_map = {}

    aliases = {
        "superclass": "super_class",
        "superClass": "super_class",
        "cellClass": "cell_class",
        "cellclass": "cell_class",
        "cellType": "cell_type",
        "celltype": "cell_type",
        "rootId": "root_id",
        "rootID": "root_id",
        "hemilineage": "hemilineage",
        "hemilineage_name": "hemilineage",
        "neurotransmitter": "nt_type",
        "nt": "nt_type",
        "nerve_entry": "nerve",
        "nerve_exit": "nerve",
    }

    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            rename_map[source] = target

    if rename_map:
        df = df.rename(
            columns=rename_map
        )

    return df


def normalize_cell_type_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize cell type values without changing their meaning."""

    if "cell_type" not in df.columns:
        return df

    df["cell_type"] = (
        df["cell_type"]
        .astype(str)
        .str.strip()
    )

    return df


def filter_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Find neurons belonging to candidate DN cell types."""

    if "cell_type" not in df.columns:
        raise ValueError(
            "Detailed annotation file does not contain "
            "'cell_type'.\n\n"
            "This usually means the wrong annotation file "
            "was supplied."
        )

    result = df[
        df["cell_type"].isin(
            DN_CANDIDATES
        )
    ].copy()

    return result


def print_missing_columns(
    df: pd.DataFrame,
) -> None:
    """Print useful annotation coverage."""

    print()
    print("=" * 70)
    print("ANNOTATION COVERAGE")
    print("=" * 70)

    for column in ANNOTATION_COLUMNS:
        status = (
            "YES"
            if column in df.columns
            else "NO"
        )

        print(
            f"{status:>3}  {column}"
        )


def print_candidate_summary(
    candidates: pd.DataFrame,
) -> None:
    """Print detailed candidate DN annotations."""

    print()
    print("=" * 70)
    print("DN CANDIDATE ANNOTATIONS")
    print("=" * 70)

    if candidates.empty:
        print("No candidate DN types found.")
        return

    preferred = [
        "cell_type",
        "flow",
        "super_class",
        "cell_class",
        "side",
        "nerve",
        "hemilineage",
        "nt_type",
        "hemibrain_type",
        "morphology_group",
    ]

    available = [
        column
        for column in preferred
        if column in candidates.columns
    ]

    summary = (
        candidates[
            available
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                column
                for column in [
                    "cell_type",
                    "side",
                ]
                if column in available
            ]
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )


def print_function_groups(
    candidates: pd.DataFrame,
) -> None:
    """Group candidate DNs by broad annotation."""

    print()
    print("=" * 70)
    print("FUNCTIONAL / ANATOMICAL GROUPS")
    print("=" * 70)

    if candidates.empty:
        return

    for column in [
        "flow",
        "super_class",
        "cell_class",
        "nerve",
        "hemilineage",
        "nt_type",
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

        for value, count in values.items():
            print(
                f"  {value}: {count}"
            )


def print_candidate_table(
    candidates: pd.DataFrame,
) -> None:
    """Print compact candidate table."""

    print()
    print("=" * 70)
    print("COMPACT DN TABLE")
    print("=" * 70)

    if candidates.empty:
        return

    columns = [
        "cell_type",
        "flow",
        "super_class",
        "cell_class",
        "side",
        "nerve",
        "hemilineage",
        "nt_type",
        "hemibrain_type",
    ]

    available = [
        column
        for column in columns
        if column in candidates.columns
    ]

    table = (
        candidates[
            available
        ]
        .drop_duplicates()
    )

    sort_columns = [
        column
        for column in [
            "cell_type",
            "side",
        ]
        if column in available
    ]

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
            "Analyze detailed FlyWire annotations "
            "for DN candidates from the visual "
            "connectome simulation."
        )
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help=(
            "Path to detailed neuron annotation "
            "TSV/CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/processed/"
            "dn_candidate_annotations.csv"
        ),
        help="Output CSV path.",
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

    annotations = normalize_cell_type_values(
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
        "Use the detailed FlyWire annotations "
        "to identify which candidate DNs are "
        "descending neurons and what anatomical "
        "class they belong to."
    )
    print(
        "Do not assign JUMP / WAIT labels yet."
    )


if __name__ == "__main__":
    main()