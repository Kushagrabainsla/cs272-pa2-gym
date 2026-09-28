"""Self-tests for PA2, run with `python myrunner.py --check`.

The environment checks mirror the assignment's requirements one by one. The
agent checks compare it against hand-computed values and against an independent
one-step SARSA, so a wrong trace update cannot hide behind a good-looking curve.
"""

from __future__ import annotations

import re
import subprocess
import sys
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from myagent import ACCUMULATING, REPLACING, RandomAgent, SarsaLambdaAgent, argmax_action
from myrunner import ENV_ID, HYPERPARAMS, make_env

ROOT = Path(__file__).parent


# --------------------------------------------------------------- environment

def check_gymnasium_api() -> None:
    check_env(make_env().unwrapped)
    assert gym.spec(ENV_ID).max_episode_steps == 100, "register() must pass max_episode_steps"


def check_space_sizes() -> None:
    env = make_env()
    assert isinstance(env.observation_space, spaces.Discrete) and 20 <= env.observation_space.n <= 500
    assert isinstance(env.action_space, spaces.Discrete) and 2 <= env.action_space.n <= 10


def check_all_randomness_from_np_random() -> None:
    source = (ROOT / "myenv.py").read_text()
    assert not re.search(r"^\s*(import random|from random)", source, re.M), "myenv imports `random`"
    assert not re.search(r"(?<!self\.)\bnp\.random\b|numpy\.random", source), "myenv uses bare numpy.random"


def check_stochastic() -> None:
    env = make_env()
    outcomes = set()
    for seed in range(40):
        env.reset(seed=seed)
        outcomes.add(env.step(1)[0])
    assert len(outcomes) > 1, "the same action in the same state always gave the same result"


def check_termination_and_truncation() -> None:
    env = make_env()
    env.reset(seed=0)
    steps, truncated = 0, False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(0)  # always "up": never delivers
        steps += 1
        assert not terminated
    assert steps == 100, f"TimeLimit should truncate after 100 steps, got {steps}"

    raw = make_env().unwrapped  # without the wrapper the env itself must never truncate
    raw.reset(seed=0)
    assert not any(raw.step(a % 4)[3] for a in range(500)), "env set truncated by hand"

    env = make_env()
    trained = SarsaLambdaAgent(env, lam=0.6, total_epi=300, seed=1, **HYPERPARAMS)
    trained.learn()
    _, reached = trained.best_run()
    assert reached, "a trained greedy policy never reached a terminal state"


def check_seeding_across_processes() -> None:
    snippet = (
        "import sys; sys.dont_write_bytecode = True; import gymnasium as gym, myenv; "
        "e = gym.make('cs272/CourierRoute-v0'); o, _ = e.reset(seed=42); t = [o]\n"
        "for i in range(40): t.append(e.step(i % 4)[:3])\n"
        "print(t)"
    )
    runs = [subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, cwd=ROOT).stdout
            for _ in range(2)]
    assert runs[0] and runs[0] == runs[1], "same seed gave different trajectories in fresh processes"


def check_render() -> None:
    env = make_env(render_mode="ansi")
    env.reset(seed=0)
    text = env.unwrapped.render()
    assert isinstance(text, str) and "@" in text and "C" in text and "D" in text
    assert make_env().unwrapped.render() is None


def check_docs_are_in_sync() -> None:
    env_doc = (ROOT / "env.md").read_text().strip()
    assert env_doc in (ROOT / "README.md").read_text(), "README.md must contain env.md verbatim"


# --------------------------------------------------------------------- agent

class ScriptedEnv(gym.Env):
    """A fixed script of (observation, reward, terminated, truncated) steps, for hand-computed tests."""

    def __init__(self, n_obs: int, script: list[tuple[int, float, bool, bool]]):
        self.observation_space = spaces.Discrete(n_obs)
        self.action_space = spaces.Discrete(1)
        self.script, self.t = script, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return 0, {}

    def step(self, action):
        obs, reward, terminated, truncated = self.script[self.t]
        self.t += 1
        return obs, reward, terminated, truncated, {}


def learn_scripted(script, n_obs, episodes=1, **kwargs) -> np.ndarray:
    params = dict(gamma=1.0, alpha=0.5, eps=0.0, init_val=0.0, total_epi=episodes, seed=0) | kwargs
    agent = SarsaLambdaAgent(ScriptedEnv(n_obs, script), **params)
    agent.learn()
    return agent.q


def check_trace_reaches_earlier_states() -> None:
    # 0 -> 1 -> terminal, reward 1 on the last step. One episode gives
    # Q(1) = alpha and Q(0) = alpha * (gamma * lambda), i.e. traces carry the
    # final error one step back, scaled by lambda.
    chain = [(1, 0.0, False, False), (2, 1.0, True, False)]
    for lam in (0.0, 0.5, 1.0):
        q = learn_scripted(chain, 3, lam=lam)
        assert np.isclose(q[1, 0], 0.5) and np.isclose(q[0, 0], 0.5 * lam), f"lambda={lam}: {q[:, 0]}"


def check_traces_are_episodic() -> None:
    # Same chain, two episodes, lambda = gamma = 1 so nothing decays on its own.
    # Episode 1 leaves Q = (0.5, 0.5). Episode 2 has delta = 0, then 0.5, so with
    # fresh traces Q = (0.75, 0.75); traces left over from episode 1 would give
    # the last state a larger share and Q(1) = 1.0.
    chain = [(1, 0.0, False, False), (2, 1.0, True, False)]
    q = learn_scripted(chain, 3, episodes=2, lam=1.0)
    assert np.allclose(q[:2, 0], 0.75), f"traces leaked across episodes: {q[:2, 0]}"


def check_accumulating_vs_replacing() -> None:
    # The same pair is visited twice, then reward 1. Accumulating traces give
    # it E = lambda + 1, replacing traces cap it at 1.
    loop = [(0, 0.0, False, False), (1, 1.0, True, False)]
    assert np.isclose(learn_scripted(loop, 2, lam=0.5, trace=ACCUMULATING)[0, 0], 0.5 * 1.5)
    assert np.isclose(learn_scripted(loop, 2, lam=0.5, trace=REPLACING)[0, 0], 0.5 * 1.0)


def check_truncation_bootstraps_but_termination_does_not() -> None:
    # One step from state 0 to state 1 with reward 0; Q starts at 1, gamma 0.5.
    # Truncated: S' is real, delta = 0 + 0.5 * 1 - 1 = -0.5. Terminated: delta = -1.
    kwargs = dict(gamma=0.5, alpha=1.0, init_val=1.0, lam=0.0)
    truncated = learn_scripted([(1, 0.0, False, True)], 2, **kwargs)
    terminated = learn_scripted([(1, 0.0, True, False)], 2, **kwargs)
    assert np.isclose(truncated[0, 0], 0.5) and np.isclose(terminated[0, 0], 0.0)
    assert np.isclose(truncated[1, 0], 1.0), "truncated state must keep its value"
    assert np.isclose(terminated[1, 0], 0.0), "terminal state must be zero"


def reference_one_step_sarsa(env, gamma, alpha, eps, episodes, init_val, seed) -> np.ndarray:
    """Textbook one-step SARSA, no traces. Same random-number order as the agent."""
    rng = np.random.default_rng(seed)
    q = np.full((env.observation_space.n, env.action_space.n), float(init_val))

    def act(s):
        return int(rng.integers(q.shape[1])) if rng.random() < eps else argmax_action(q[s], rng)

    env_seed = int(rng.integers(2**31 - 1))
    for _ in range(episodes):
        s, _ = env.reset(seed=env_seed)
        env_seed = None
        a = act(s)
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            if terminated:
                q[s2, :] = 0.0
                q[s, a] += alpha * (r - q[s, a])
                break
            a2 = act(s2)
            q[s, a] += alpha * (r + gamma * q[s2, a2] - q[s, a])
            if truncated:
                break
            s, a = s2, a2
    return q


def check_lambda_zero_is_one_step_sarsa() -> None:
    params = dict(HYPERPARAMS, total_epi=300, seed=5)
    agent = SarsaLambdaAgent(make_env(), lam=0.0, **params)
    agent.learn()
    reference = reference_one_step_sarsa(make_env(), params["gamma"], params["alpha"], params["eps"],
                                         params["total_epi"], params["init_val"], params["seed"])
    assert np.allclose(agent.q, reference, atol=1e-12, rtol=0), "lambda=0 differs from one-step SARSA"
    other = SarsaLambdaAgent(make_env(), lam=0.9, **params)
    other.learn()
    assert not np.allclose(other.q, reference), "lambda=0.9 should differ from one-step SARSA"


def check_learning_curve_is_reproducible() -> None:
    curves = []
    for _ in range(2):
        agent = SarsaLambdaAgent(make_env(), lam=0.6, total_epi=150, seed=9, **HYPERPARAMS)
        curves.append(agent.learn())
    assert curves[0] == curves[1], "same seed, different learning curve"


def check_agent_is_environment_independent() -> None:
    assert "myenv" not in (ROOT / "myagent.py").read_text(), "myagent.py must not reference myenv"
    for env_id, states, actions in (("FrozenLake-v1", 16, 4), ("Taxi-v3", 500, 6)):
        env = gym.make(env_id)
        agent = SarsaLambdaAgent(env, total_epi=30, seed=0)
        assert len(agent.learn()) == 30 and agent.q.shape == (states, actions)
        episode, _ = agent.best_run()
        assert episode and agent.calc_return(episode) == sum(r for *_, r in episode)


def check_env_is_worth_learning() -> None:
    env = make_env()
    env.reset(seed=8)  # RandomAgent does not seed the env itself
    random_return = float(np.mean(RandomAgent(env, total_epi=300, seed=8).learn()))
    agent = SarsaLambdaAgent(make_env(), lam=0.9, total_epi=1000, seed=8, **HYPERPARAMS)
    trained_return = float(np.mean(agent.learn()[-100:]))
    assert trained_return > random_return + 5, f"trained {trained_return:.2f} vs random {random_return:.2f}"


CHECKS = [
    check_gymnasium_api, check_space_sizes, check_all_randomness_from_np_random, check_stochastic,
    check_termination_and_truncation, check_seeding_across_processes, check_render, check_docs_are_in_sync,
    check_trace_reaches_earlier_states, check_traces_are_episodic, check_accumulating_vs_replacing,
    check_truncation_bootstraps_but_termination_does_not, check_lambda_zero_is_one_step_sarsa,
    check_learning_curve_is_reproducible, check_agent_is_environment_independent, check_env_is_worth_learning,
]


def run_all() -> None:
    failures = 0
    for check in CHECKS:
        name = check.__name__.removeprefix("check_").replace("_", " ")
        try:
            check()
            print(f"[PASS] {name}")
        except Exception:  # report every failure, not just the first
            failures += 1
            print(f"[FAIL] {name}\n{traceback.format_exc()}")
    if failures:
        raise SystemExit(f"{failures} of {len(CHECKS)} checks failed")
    print(f"All {len(CHECKS)} checks passed.")
