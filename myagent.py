"""Environment-independent tabular SARSA(lambda) with eligibility traces."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return a uniformly random action among all actions tied for maximum."""
    values = np.asarray(values)
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(best))


class SarsaLambdaAgent:
    def __init__(self, env: gym.Env, gamma: float = 0.99, alpha: float = 0.05,
                 eps: float = 0.1, lam: float = 0.9, trace: str = ACCUMULATING,
                 total_epi: int = 5_000, init_val: float = 1.0,
                 seed: int | None = None) -> None:
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError("trace must be 'accumulating' or 'replacing'")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0 or not 0.0 <= eps <= 1.0:
            raise ValueError("gamma, lambda, and eps must be in [0, 1]")
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("observation_space must be Discrete")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("action_space must be Discrete")
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma, self.alpha, self.eps, self.lam = gamma, alpha, eps, lam
        self.trace, self.total_epi, self.init_val = trace, total_epi, init_val
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        return np.full((self.n_states, self.n_actions), float(init_val), dtype=np.float64)

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        if exploration and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return argmax_action(self.q[int(state)], self.rng)

    def learn(self) -> list[float]:
        returns: list[float] = []
        for episode_number in range(self.total_epi):
            reset_seed = None if self.seed is None else self.seed + episode_number
            state, _ = self.env.reset(seed=reset_seed)
            state = int(state)
            action = self.eps_greedy(state)
            traces = np.zeros_like(self.q)
            total_return = 0.0
            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state, reward = int(next_state), float(reward)
                total_return += reward
                if terminated:
                    delta = reward - self.q[state, action]
                    next_action = None
                    # The terminal observation has no future value and must
                    # never be updated as a current state.
                    self.q[next_state, :] = 0.0
                else:
                    next_action = self.eps_greedy(next_state)
                    delta = reward + self.gamma * self.q[next_state, next_action] - self.q[state, action]
                if self.trace == REPLACING:
                    traces[state, action] = 1.0
                else:
                    traces[state, action] += 1.0
                self.q += self.alpha * delta * traces
                traces *= self.gamma * self.lam
                if terminated or truncated:
                    break
                state, action = next_state, int(next_action)
            returns.append(total_return)
        return returns

    def best_run(self, max_steps: int = 300) -> tuple[list[tuple[int, int, float]], bool]:
        state, _ = self.env.reset(seed=None if self.seed is None else self.seed + 10_000_000)
        state = int(state)
        episode: list[tuple[int, int, float]] = []
        for _ in range(max_steps):
            action = self.eps_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, float(reward)))
            state = int(next_state)
            if terminated or truncated:
                return episode, bool(terminated)
        return episode, False

    def calc_return(self, episode: list[tuple[Any, Any, float]], discounted: bool = False) -> float:
        if not discounted:
            return float(sum(float(step[2]) for step in episode))
        return float(sum((self.gamma ** i) * float(step[2]) for i, step in enumerate(episode)))


class RandomAgent(SarsaLambdaAgent):
    """Uniform-random baseline used by the report."""

    def learn(self) -> list[float]:
        returns = []
        for episode_number in range(self.total_epi):
            reset_seed = None if self.seed is None else self.seed + episode_number
            self.env.reset(seed=reset_seed)
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
        return returns
