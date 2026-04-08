"""Neural network utilities for RL agents."""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class PolicyNetwork(nn.Module):
    """Simple feedforward policy network for discrete actions.

    Outputs action probabilities for discrete action spaces.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """Initialize policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = state_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (action logits)
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Action logits (batch_size, action_dim)
        """
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities.

        Args:
            state: State tensor

        Returns:
            Action probabilities (batch_size, action_dim)
        """
        logits = self.forward(state)
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Simple feedforward value network.

    Estimates state value V(s) or state-action value Q(s,a).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128]):
        """Initialize value network.

        Args:
            input_dim: Dimension of input (state_dim for V, state_dim + action_dim for Q)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Value estimates (batch_size, 1)
        """
        return self.network(x)


def to_tensor(x: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert numpy array to torch tensor.

    Args:
        x: Numpy array
        device: Device to put tensor on

    Returns:
        Torch tensor
    """
    return torch.FloatTensor(x).to(device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array.

    Args:
        x: Torch tensor

    Returns:
        Numpy array
    """
    return x.detach().cpu().numpy()
