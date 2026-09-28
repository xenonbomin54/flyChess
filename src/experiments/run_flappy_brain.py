from __future__ import annotations

from pathlib import Path

from src.brain.simulation import NeuralSimulation
from src.brain.visual_input import (
    FlappyBirdState,
    VisualEncoder,
)
from src.connectome.graph import (
    attach_cell_types,
    build_graph,
)
from src.connectome.loader import (
    load_cell_types,
    load_connectome,
)


CONNECTOME_PATH = Path(
    "data/raw/connections_princeton.csv.gz"
)

CELL_TYPES_PATH = Path(
    "data/raw/consolidated_cell_types.csv.gz"
)


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


def load_brain() -> tuple:
    """Load the connectome and cell type information."""

    print("=" * 70)
    print("LOADING CONNECTOME")
    print("=" * 70)

    connections = load_connectome(
        str(CONNECTOME_PATH)
    )

    cell_types = load_cell_types(
        str(CELL_TYPES_PATH)
    )

    print(
        f"Connection rows: {len(connections):,}"
    )

    graph = build_graph(
        connections
    )

    graph = attach_cell_types(
        graph,
        cell_types,
    )

    print(
        f"Nodes: {graph.number_of_nodes():,}"
    )

    print(
        f"Edges: {graph.number_of_edges():,}"
    )

    return graph


def get_visual_neurons(
    graph,
    visual_type: str,
    limit: int = 30,
) -> list[int]:
    """Get neurons belonging to one T4/T5 visual type."""

    neurons = []

    for neuron_id, data in graph.nodes(
        data=True
    ):
        cell_type = data.get(
            "cell_type",
            "",
        )

        if cell_type == visual_type:
            neurons.append(
                neuron_id
            )

            if len(neurons) >= limit:
                break

    return neurons


def convert_visual_activity_to_signal(
    activity: float,
) -> float:
    """
    Convert visual activity into a neural stimulation signal.

    The connectome simulation uses a threshold of 0.3.
    Therefore active visual features are mapped into
    the range [0.3, 1.0].
    """

    if activity <= 0:
        return 0.0

    return 0.3 + (
        0.7 * min(
            activity,
            1.0,
        )
    )


def stimulate_visual_input(
    simulation: NeuralSimulation,
    graph,
    visual_input: dict[str, float],
) -> dict[str, float]:
    """
    Convert visual feature activity into connectome stimulation.

    Only active visual channels are stimulated.
    """

    stimulated = {}

    for visual_type, activity in visual_input.items():
        if activity <= 0:
            continue

        neurons = get_visual_neurons(
            graph,
            visual_type,
            limit=30,
        )

        if not neurons:
            continue

        signal = (
            convert_visual_activity_to_signal(
                activity
            )
        )

        simulation.stimulate_many(
            neurons,
            signal=signal,
        )

        stimulated[
            visual_type
        ] = signal

    return stimulated


def run_brain(
    graph,
    state: FlappyBirdState,
    steps: int = 12,
) -> tuple[
    list[list[int]],
    dict[str, float],
    dict[str, float],
]:
    """Run the connectome from a Flappy Bird visual state."""

    encoder = VisualEncoder()

    visual_input = encoder.encode(
        state
    )

    simulation = NeuralSimulation(
        graph,
        threshold=0.3,
        decay=0.9,
    )

    stimulated = stimulate_visual_input(
        simulation,
        graph,
        visual_input,
    )

    history = simulation.run(
        steps=steps
    )

    return (
        history,
        visual_input,
        stimulated,
    )


def summarize_history(
    graph,
    history: list[list[int]],
) -> dict[str, int]:
    """Summarize firing activity by annotated neuron type."""

    counts: dict[str, int] = {}

    for fired in history:
        for neuron_id in fired:
            neuron_type = graph.nodes[
                neuron_id
            ].get(
                "cell_type",
                "Unknown",
            )

            counts[neuron_type] = (
                counts.get(
                    neuron_type,
                    0,
                )
                + 1
            )

    return dict(
        sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def main() -> None:
    graph = load_brain()

    state = FlappyBirdState(
        bird_x=200,
        bird_y=250,
        pipe_x=400,
        pipe_gap_y=300,
        pipe_width=80,
        pipe_gap_height=150,
        screen_width=800,
        screen_height=600,
    )

    print()
    print("=" * 70)
    print("FLAPPY BIRD VISUAL INPUT")
    print("=" * 70)

    encoder = VisualEncoder()

    visual_input = encoder.encode(
        state
    )

    for visual_type in VISUAL_TYPES:
        print(
            f"{visual_type}: "
            f"{visual_input[visual_type]:.4f}"
        )

    history, _, stimulated = run_brain(
        graph,
        state,
    )

    print()
    print("=" * 70)
    print("STIMULATED VISUAL NEURONS")
    print("=" * 70)

    for visual_type, signal in stimulated.items():
        print(
            f"{visual_type}: "
            f"signal={signal:.4f} "
            f"(30 neurons)"
        )

    print()
    print("=" * 70)
    print("NEURAL SIMULATION")
    print("=" * 70)

    for step, fired in enumerate(
        history
    ):
        print(
            f"Step {step:02d}: "
            f"{len(fired):,} neurons fired"
        )

    summary = summarize_history(
        graph,
        history,
    )

    print()
    print("=" * 70)
    print("TOP NEURAL OUTPUTS")
    print("=" * 70)

    if not summary:
        print(
            "No neurons fired."
        )
    else:
        for neuron_type, count in list(
            summary.items()
        )[:20]:
            print(
                f"{neuron_type:<15} "
                f"{count:>6}"
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
