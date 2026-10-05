"""Unit tests for SarsaLambdaAgent and RandomAgent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import myenv  # registers environment
from myagent import ACCUMULATING, REPLACING, RandomAgent, SarsaLambdaAgent, argmax_action


def test_argmax_action_tie_breaking():
    rng = np.random.default_rng(42)
    # Uniform values: all actions are tied
    values = np.zeros(4)
    actions = [argmax_action(values, rng) for _ in range(1000)]
    counts = np.bincount(actions, minlength=4)
    # Check each action was chosen roughly uniformly (1000 / 4 = 250)
    for c in counts:
        assert 180 < c < 320


def test_argmax_action_clear_winner():
    rng = np.random.default_rng(0)
    values = np.array([0.1, 0.2, 0.9, 0.3])
    for _ in range(10):
        assert argmax_action(values, rng) == 2


def test_eps_greedy_pure_greedy():
    env = gym.make("cs272/CourierRoute-v0")
    agent = SarsaLambdaAgent(env, eps=0.0, seed=0)
    agent.q[0, :] = [1.0, 5.0, 2.0, 0.0]
    # Under eps=0.0, action 1 should always be chosen
    for _ in range(20):
        assert agent.eps_greedy(0, exploration=True) == 1
    env.close()


def test_eps_greedy_pure_exploratory():
    env = gym.make("cs272/CourierRoute-v0")
    agent = SarsaLambdaAgent(env, eps=1.0, seed=42)
    actions = [agent.eps_greedy(0, exploration=True) for _ in range(400)]
    counts = np.bincount(actions, minlength=4)
    for c in counts:
        assert c > 0
    env.close()


def test_epsilon_decay():
    env = gym.make("cs272/CourierRoute-v0")
    agent = SarsaLambdaAgent(env, eps=0.5, eps_decay=0.9, min_eps=0.1, total_epi=10, seed=0)
    assert agent.current_eps == 0.5
    agent.learn()
    # 0.5 * (0.9 ** 10) = 0.174
    assert np.isclose(agent.current_eps, 0.5 * (0.9 ** 10), atol=1e-4)
    env.close()


def test_get_policy_and_state_values():
    env = gym.make("cs272/CourierRoute-v0")
    agent = SarsaLambdaAgent(env, seed=0)
    agent.q.fill(0.0)
    agent.q[0, 1] = 10.0
    agent.q[1, 2] = 20.0

    policy = agent.get_policy()
    v_table = agent.get_state_values()

    assert policy.shape == (env.observation_space.n,)
    assert v_table.shape == (env.observation_space.n,)
    assert policy[0] == 1
    assert policy[1] == 2
    assert v_table[0] == 10.0
    assert v_table[1] == 20.0
    env.close()


def test_agent_environment_independence():
    for env_name, n_s, n_a in (("FrozenLake-v1", 16, 4), ("Taxi-v3", 500, 6)):
        env = gym.make(env_name)
        agent = SarsaLambdaAgent(env, total_epi=20, seed=0)
        returns = agent.learn()
        assert len(returns) == 20
        assert agent.q.shape == (n_s, n_a)
        episode, _ = agent.best_run(max_steps=50)
        assert isinstance(episode, list)
        env.close()


def test_calc_return_discounted_and_undiscounted():
    env = gym.make("cs272/CourierRoute-v0")
    agent = SarsaLambdaAgent(env, gamma=0.5)
    episode = [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 4.0)]
    undisc = agent.calc_return(episode, discounted=False)
    disc = agent.calc_return(episode, discounted=True)
    assert undisc == 7.0
    # 1.0 + 0.5 * 2.0 + 0.25 * 4.0 = 1.0 + 1.0 + 1.0 = 3.0
    assert disc == 3.0
    env.close()


def test_random_agent_execution():
    env = gym.make("cs272/CourierRoute-v0")
    random_agent = RandomAgent(env, total_epi=10, seed=42)
    returns = random_agent.learn()
    assert len(returns) == 10
    # All returns should be negative or small
    assert np.mean(returns) < 5.0
    env.close()
