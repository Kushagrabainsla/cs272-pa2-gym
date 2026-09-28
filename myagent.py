"""Task 2: tabular SARSA(lambda) with eligibility traces.

The agent is environment-independent. It reads its table sizes off the two
Discrete spaces of whatever environment it is given, and it never interprets a
state number.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the largest value, breaking ties uniformly at random.

    Ties are not an edge case here. The table starts uniform, so on the first
    visit to a state every action is tied, and a plain np.argmax would commit
    every state in the table to action 0.

    Args:
        values: the q-values of one state, shape (n_actions,)
        rng: the agent's random generator

    Returns:
        int: an action
    """
    best = np.flatnonzero(values == np.max(values))
    return int(rng.choice(best))


class SarsaLambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.05,
        eps: float = 0.1,
        lam: float = 0.9,
        trace: str = ACCUMULATING,
        total_epi: int = 5_000,
        init_val: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            env: any tabular gym environment. Both spaces are Discrete.
            gamma: discount factor.
            alpha: learning rate.
            eps: exploration rate for a plain (non-decaying) epsilon-greedy.
            lam: the lambda of SARSA(lambda), in [0, 1]. At 0 this reduces to
                ordinary one-step SARSA.
            trace: "replacing" or "accumulating".
            total_epi: number of training episodes.
            init_val: value every q(s,a) starts at.
            seed: seed for the agent's own randomness and, once at the start of
                learn(), for the environment's. Same seed, same learning curve.
        """
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError(f"unknown trace type: {trace}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0 and 0.0 <= eps <= 1.0):
            raise ValueError("gamma, lam, and eps must be in [0, 1]")

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.trace = trace
        self.total_epi = total_epi
        self.init_val = init_val
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        """Build the q table, shape (n_states, n_actions), filled with init_val."""
        return np.full((self.n_states, self.n_actions), float(init_val), dtype=np.float64)

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """Epsilon-greedy action selection over the current q table.

        Args:
            state: the current state
            exploration: explore with probability eps if True; act greedily if
                False. The greedy path is what best_run uses.

        Returns:
            int: an action
        """
        if exploration and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return argmax_action(self.q[int(state)], self.rng)

    def learn(self) -> list[float]:
        """Run SARSA(lambda) for self.total_epi episodes, updating self.q.

        Returns:
            list[float]: the undiscounted return of each training episode, in
            order. myrunner.py plots these.
        """
        returns: list[float] = []
        env_seed = None
        if self.seed is not None:
            # Derived from the agent's stream, not equal to self.seed: two
            # generators built from the same integer would emit identical
            # numbers, tying exploration to the environment's noise.
            env_seed = int(self.rng.integers(2**31 - 1))

        for _ in range(self.total_epi):
            state, _ = self.env.reset(seed=env_seed)
            env_seed = None  # seed once; later resets continue the same stream
            state = int(state)
            action = self.eps_greedy(state)
            traces = np.zeros_like(self.q)  # traces are episodic
            total_return = 0.0

            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state, reward = int(next_state), float(reward)
                total_return += reward

                if terminated:
                    # No next state exists. Q(terminal, .) = 0 and is never
                    # updated: its traces stay zero because it is never a
                    # current state.
                    self.q[next_state, :] = 0.0
                    next_action = None
                    delta = reward - self.q[state, action]
                else:
                    # Truncation is not termination: next_state is a real state.
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
                state, action = next_state, next_action
            returns.append(total_return)
        return returns

    def best_run(self, max_steps: int = 300, reset_seed: int | None = None) -> tuple[list[tuple[int, int, float]], bool]:
        """Generate one greedy episode under the learned q table, for the report.

        Args:
            max_steps: give up after this many steps.
            reset_seed: optional seed for the environment reset, so the same
                episode can be replayed (for example to render it).

        Returns:
            tuple[
                list[tuple[int,int,float]]: the episode, as [(s, a, r), ...]
                bool: True if it reached a terminal state, False if it ran out
            ]
        """
        state, _ = self.env.reset(seed=reset_seed)
        state = int(state)
        episode: list[tuple[int, int, float]] = []
        for _ in range(max_steps):
            action = self.eps_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, float(reward)))
            state = int(next_state)
            if terminated:
                return episode, True
            if truncated:
                return episode, False
        return episode, False

    def calc_return(self, episode: list[tuple[Any, Any, float]], discounted: bool = False) -> float:
        """Return of an episode given as [(s, a, r), ...]."""
        if not discounted:
            return float(sum(float(step[2]) for step in episode))
        return float(sum((self.gamma ** i) * float(step[2]) for i, step in enumerate(episode)))


class RandomAgent(SarsaLambdaAgent):
    """The baseline your agent has to beat. Already written; do not change it."""

    def learn(self) -> list[float]:
        returns = []
        for _ in range(self.total_epi):
            self.env.reset()
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        return returns
