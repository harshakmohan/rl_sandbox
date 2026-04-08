"""Train REINFORCE agent on CartPole-v1."""

import sys
sys.path.append('..')

from environments.base import GymEnv
from agents.reinforce import REINFORCEAgent
from utils.trainer import Trainer


env = GymEnv('CartPole-v1')

agent = REINFORCEAgent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    learning_rate=1e-3,
    gamma=0.99
)

trainer = Trainer(env, agent, log_dir="runs/reinforce")
trainer.train(n_episodes=1000)
