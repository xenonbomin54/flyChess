from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlappyBirdState:
    """Minimal Flappy Bird state used as visual input."""

    bird_x: float
    bird_y: float

    pipe_x: float
    pipe_gap_y: float

    pipe_width: float
    pipe_gap_height: float

    screen_width: float
    screen_height: float


class VisualEncoder:
    """
    Encode Flappy Bird state into coarse visual directions.

    The encoder does not decide JUMP or WAIT.
    It only converts the game state into visual features.
    """

    VISUAL_TYPES = (
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    )

    def encode(
        self,
        state: FlappyBirdState,
    ) -> dict[str, float]:
        """Convert game state into visual feature intensities."""

        horizontal_distance = (
            state.pipe_x
            - state.bird_x
        )

        vertical_distance = (
            state.pipe_gap_y
            - state.bird_y
        )

        horizontal_scale = max(
            state.screen_width,
            1.0,
        )

        vertical_scale = max(
            state.screen_height,
            1.0,
        )

        horizontal = max(
            -1.0,
            min(
                1.0,
                horizontal_distance
                / horizontal_scale,
            ),
        )

        vertical = max(
            -1.0,
            min(
                1.0,
                vertical_distance
                / vertical_scale,
            ),
        )

        features = {
            "T4a": max(
                0.0,
                -horizontal,
            ),
            "T4b": max(
                0.0,
                horizontal,
            ),
            "T4c": max(
                0.0,
                -vertical,
            ),
            "T4d": max(
                0.0,
                vertical,
            ),
            "T5a": max(
                0.0,
                -horizontal,
            ),
            "T5b": max(
                0.0,
                horizontal,
            ),
            "T5c": max(
                0.0,
                -vertical,
            ),
            "T5d": max(
                0.0,
                vertical,
            ),
        }

        return features

    def encode_to_active_types(
        self,
        state: FlappyBirdState,
        threshold: float = 0.1,
    ) -> list[str]:
        """Return visual types whose activity exceeds the threshold."""

        features = self.encode(state)

        return [
            visual_type
            for visual_type, activity in features.items()
            if activity >= threshold
        ]
