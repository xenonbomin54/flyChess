from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionSignal:
    """Continuous action signal produced by the neural output."""

    jump_score: float
    wait_score: float

    @property
    def action(self) -> str:
        """Return the stronger action."""

        if self.jump_score > self.wait_score:
            return "JUMP"

        return "WAIT"


class BrainController:
    """Convert neural output activity into Flappy Bird actions."""

    def __init__(
        self,
        jump_neurons: list[str] | None = None,
        wait_neurons: list[str] | None = None,
    ):
        self.jump_neurons = set(
            jump_neurons or []
        )

        self.wait_neurons = set(
            wait_neurons or []
        )

    def set_action_neurons(
        self,
        jump_neurons: list[str],
        wait_neurons: list[str],
    ) -> None:
        """Set DN groups used for each action."""

        self.jump_neurons = set(
            jump_neurons
        )

        self.wait_neurons = set(
            wait_neurons
        )

    def calculate_signal(
        self,
        output_vector: dict[str, float],
    ) -> ActionSignal:
        """Calculate continuous JUMP and WAIT scores."""

        jump_score = sum(
            output_vector.get(
                neuron,
                0.0,
            )
            for neuron in self.jump_neurons
        )

        wait_score = sum(
            output_vector.get(
                neuron,
                0.0,
            )
            for neuron in self.wait_neurons
        )

        return ActionSignal(
            jump_score=jump_score,
            wait_score=wait_score,
        )

    def decide(
        self,
        output_vector: dict[str, float],
    ) -> str:
        """Convert neural output into a Flappy Bird action."""

        signal = self.calculate_signal(
            output_vector
        )

        return signal.action
