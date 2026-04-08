"""Base agent interface for RL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class Agent(ABC):
    """Base class for all RL agents.

    Defines the interface that all agents must implement.
    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (discrete) or action vector (continuous)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> Any:
        """Choose an action given a state.

        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def learn(self, state: np.ndarray, action: Any, reward: float,
              next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Learn from a transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated

        Returns:
            Dictionary of metrics (e.g., loss values)
        """
        pass

    def save(self, path: str):
        """Save agent to disk."""
        pass

    def load(self, path: str):
        """Load agent from disk."""
        pass
