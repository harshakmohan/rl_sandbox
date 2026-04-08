# RL Research Framework

A minimal, modular framework for RL research — implementing algorithms from scratch and running controlled experiments.

---

## Intent

This repo is built for two things:

1. **Algorithm implementation** — implement RL algorithms from scratch to build a deep understanding of how they work
2. **Research** — run controlled experiments where agents, environments, and reward functions can be swapped independently to isolate the effect of each change

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running an Experiment

```bash
cd experiments
python train_reinforce.py
```

View training metrics:

```bash
tensorboard --logdir experiments/runs/
```

---

## Structure

```
agents/
    base.py              - Agent base class (act, learn interface)
    reinforce.py         - REINFORCE (Monte Carlo policy gradient)

environments/
    base.py              - Gymnasium environment wrapper

utils/
    networks.py          - PolicyNetwork and ValueNetwork (PyTorch)
    trainer.py           - Shared training loop
    logger.py            - Metrics tracking

experiments/
    train_reinforce.py   - REINFORCE on CartPole-v1

notebooks/
    01_getting_started   - Intro walkthrough
```

---

## How Experiments Work

Each experiment creates an env, agent, and trainer — then runs:

```python
env = GymEnv('CartPole-v1')

agent = REINFORCEAgent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    learning_rate=1e-3,
    gamma=0.99
)

trainer = Trainer(env, agent, log_dir="runs/reinforce")
trainer.train(n_episodes=1000)
```

### Swapping components

**Different agent** — instantiate a different agent class, everything else stays the same:

```python
agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
trainer = Trainer(env, agent, log_dir="runs/dqn")
```

**Different environment** — pass a different env:

```python
env = GymEnv('MountainCar-v0')
```

**Custom reward function** — pass a `reward_fn` to override the environment reward without touching the agent or env:

```python
def shaped_reward(state, action, reward, next_state, done):
    cart_pos = next_state[0]
    return reward - 0.1 * abs(cart_pos)

trainer = Trainer(env, agent, reward_fn=shaped_reward, log_dir="runs/reinforce_shaped")
```

Compare runs in TensorBoard by pointing it at the parent `runs/` directory.

---

## Adding a New Algorithm

Subclass `Agent` and implement `act()` and `learn()`:

```python
from agents.base import Agent

class MyAgent(Agent):
    def act(self, state, training=True):
        # return action given state
        ...

    def learn(self, state, action, reward, next_state, done):
        # update policy/value estimates
        # return dict of metrics to log (e.g. {'loss': 0.42})
        ...
```

The `Trainer` handles the rest — episode loop, logging, TensorBoard writes.

---

## Environment

**CartPole-v1** (Gymnasium) — the standard benchmark for discrete control:

- **State**: [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: push left or push right
- **Reward**: +1 per timestep the pole stays upright
- **Episode ends**: pole angle > 12°, cart out of bounds, or 500 steps reached
- **Solved**: mean return ≥ 475 over 100 consecutive episodes

---

## Algorithms

| Algorithm | Status | Notes |
|---|---|---|
| REINFORCE | Implemented | Monte Carlo policy gradient, on-policy |
| DQN | Planned | |
| A2C | Planned | |
| PPO | Planned | |
