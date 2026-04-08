"""Shared training loop for all RL agents."""

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Optional
from tqdm import tqdm


class Trainer:
    """Runs the training loop for any agent on any environment.

    Swap agents, environments, and reward functions independently
    to isolate the effect of each change.
    """

    def __init__(
        self,
        env,
        agent,
        reward_fn: Optional[Callable] = None,
        log_dir: str = "runs"
    ):
        """
        Args:
            env: Environment (GymEnv or compatible)
            agent: Agent (any subclass of Agent)
            reward_fn: Optional reward override.
                       Signature: f(state, action, reward, next_state, done) -> float
                       If None, uses the environment's reward directly.
            log_dir: TensorBoard log directory (use distinct names per experiment)
        """
        self.env = env
        self.agent = agent
        self.reward_fn = reward_fn
        self.writer = SummaryWriter(log_dir)

    def train(
        self,
        n_episodes: int,
        max_steps: int = 500,
        seed: int = 42,
        print_every: int = 100
    ):
        """Run training loop.

        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            print_every: Print progress every N episodes
        """
        np.random.seed(seed)

        for episode in tqdm(range(1, n_episodes + 1)):
            episode_return, episode_length, metrics = self._run_episode(max_steps)

            self.writer.add_scalar("train/episode_return", episode_return, episode)
            self.writer.add_scalar("train/episode_length", episode_length, episode)
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, episode)

            if episode % print_every == 0:
                print(f"Episode {episode:4d} | Return: {episode_return:7.2f} | Steps: {episode_length}")

        self.writer.close()
        self.env.close()

    def _run_episode(self, max_steps: int):
        state = self.env.reset()
        episode_return = 0.0
        metrics = {}
        step = 0

        for step in range(max_steps):
            action = self.agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.reward_fn is not None:
                reward = self.reward_fn(state, action, reward, next_state, done)

            step_metrics = self.agent.learn(state, action, reward, next_state, done)
            metrics.update(step_metrics)

            episode_return += reward
            state = next_state

            if done:
                break

        return episode_return, step + 1, metrics
