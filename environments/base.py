"""Environment wrappers for Gymnasium."""

import gymnasium as gym
import numpy as np
from typing import Tuple


class GymEnv:
    """Wrapper around Gymnasium environments for consistency."""

    def __init__(self, env_name: str, render_mode: str = None):
        """Initialize environment.

        Args:
            env_name: Name of Gymnasium environment (e.g., 'CartPole-v1')
            render_mode: Render mode ('human', 'rgb_array', or None)
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env_name = env_name

        # Get space dimensions
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

    def _get_state_dim(self) -> int:
        """Get state space dimension."""
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return self.env.observation_space.shape[0]
        else:
            raise NotImplementedError(f"Observation space {type(self.env.observation_space)} not supported")

    def _get_action_dim(self) -> int:
        """Get action space dimension."""
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        else:
            raise NotImplementedError(f"Action space {type(self.env.action_space)} not supported")

    def reset(self) -> np.ndarray:
        """Reset environment.

        Returns:
            Initial state observation
        """
        state, info = self.env.reset()
        return state

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()
