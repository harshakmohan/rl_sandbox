"""Training logger and visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict


class Logger:
    """Simple logger for tracking training metrics."""

    def __init__(self):
        """Initialize logger."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.episode = 0

    def log(self, **kwargs):
        """Log metrics for current episode.

        Args:
            **kwargs: Metric name-value pairs
        """
        for key, value in kwargs.items():
            self.metrics[key].append(value)
        self.episode += 1

    def get_metric(self, key: str) -> List[float]:
        """Get logged values for a metric.

        Args:
            key: Metric name

        Returns:
            List of values
        """
        return self.metrics.get(key, [])

    def print_summary(self, window: int = 100):
        """Print summary of recent performance.

        Args:
            window: Window size for averaging
        """
        if self.episode == 0:
            return

        episode_returns = self.get_metric('episode_return')
        if episode_returns:
            recent_returns = episode_returns[-window:]
            print(f"Episode {self.episode} | "
                  f"Avg Return (last {len(recent_returns)}): {np.mean(recent_returns):.2f} | "
                  f"Std: {np.std(recent_returns):.2f}")

    def plot_metrics(self, metrics: List[str] = None, window: int = 10):
        """Plot training metrics.

        Args:
            metrics: List of metric names to plot (None = plot all)
            window: Window size for moving average
        """
        if metrics is None:
            metrics = list(self.metrics.keys())

        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = self.get_metric(metric)
            if not values:
                continue

            episodes = range(1, len(values) + 1)

            # Plot raw values
            ax.plot(episodes, values, alpha=0.3, label='Raw')

            # Plot moving average
            if len(values) >= window:
                moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
                ax.plot(range(window, len(values) + 1), moving_avg, label=f'{window}-episode MA')

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
