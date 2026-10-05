"""Unit tests for Courier Route Gym environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import myenv
from myenv import Action, CourierRouteEnv


def test_env_registered():
    env = gym.make("cs272/CourierRoute-v0")
    assert isinstance(env.unwrapped, CourierRouteEnv)
    env.close()


def test_gymnasium_api_compliance():
    env = CourierRouteEnv()
    check_env(env)
    env.close()


def test_space_definitions():
    env = CourierRouteEnv()
    assert isinstance(env.observation_space, spaces.Discrete)
    assert env.observation_space.n == 98
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 4
    env.close()


def test_action_enum_values():
    assert Action.UP == 0
    assert Action.RIGHT == 1
    assert Action.DOWN == 2
    assert Action.LEFT == 3
    assert len(Action) == 4


def test_state_encode_decode_roundtrip():
    for stage in (0, 1):
        for row in range(CourierRouteEnv.SIZE):
            for col in range(CourierRouteEnv.SIZE):
                state = CourierRouteEnv.encode(row, col, stage)
                assert 0 <= state < CourierRouteEnv.N_STATES
                dec_row, dec_col, dec_stage = CourierRouteEnv.decode(state)
                assert (dec_row, dec_col, dec_stage) == (row, col, stage)


def test_coordinate_helpers():
    assert CourierRouteEnv.is_valid_coord(0, 0)
    assert CourierRouteEnv.is_valid_coord(6, 6)
    assert not CourierRouteEnv.is_valid_coord(-1, 0)
    assert not CourierRouteEnv.is_valid_coord(0, 7)
    assert not CourierRouteEnv.is_valid_coord(7, 7)

    assert CourierRouteEnv.manhattan_distance(0, 0, 3, 3) == 6
    assert CourierRouteEnv.manhattan_distance(3, 3, 6, 6) == 6
    assert CourierRouteEnv.manhattan_distance(0, 0, 6, 6) == 12


def test_reset_state_and_info():
    env = CourierRouteEnv()
    obs, info = env.reset(seed=42)
    assert obs == 0
    assert info == {"row": 0, "col": 0, "stage": 0, "has_package": False}
    env.close()


def test_deterministic_seeding():
    env1 = CourierRouteEnv()
    env2 = CourierRouteEnv()
    o1, _ = env1.reset(seed=123)
    o2, _ = env2.reset(seed=123)
    assert o1 == o2

    actions = [Action.DOWN, Action.RIGHT, Action.DOWN, Action.RIGHT]
    for a in actions:
        res1 = env1.step(a)[:3]
        res2 = env2.step(a)[:3]
        assert res1 == res2
    env1.close()
    env2.close()


def test_wall_clipping():
    env = CourierRouteEnv()
    env.reset(seed=0)
    # Attempt moving UP from (0, 0)
    obs, reward, terminated, truncated, info = env.step(Action.UP)
    assert info["row"] == 0 and info["col"] in (0, 1)  # row stays 0 even if slipped right/left
    assert reward == -0.02
    assert not terminated
    assert not truncated
    env.close()


def test_checkpoint_pickup_and_delivery():
    env = CourierRouteEnv()
    env.reset()
    # Force state right before checkpoint
    env.row, env.col, env.stage = 3, 2, 0
    # Step into checkpoint without slipping by setting deterministic noise or repeatedly
    while env.stage == 0:
        env.row, env.col, env.stage = 3, 2, 0
        obs, reward, terminated, _, info = env.step(Action.RIGHT)
        if (info["row"], info["col"]) == (3, 3):
            assert info["stage"] == 1
            assert info["has_package"] is True
            assert np.isclose(reward, 0.98)  # -0.02 + 1.0
            assert not terminated

    # Now step to destination from (6, 5)
    while not terminated:
        env.row, env.col, env.stage = 6, 5, 1
        obs, reward, terminated, _, info = env.step(Action.RIGHT)
        if (info["row"], info["col"]) == (6, 6):
            assert np.isclose(reward, 9.98)  # -0.02 + 10.0
            assert terminated is True
            assert obs == 97  # 49 + 48
    env.close()


def test_dock_before_pickup_does_not_terminate():
    env = CourierRouteEnv()
    env.reset()
    env.row, env.col, env.stage = 6, 6, 0  # on dock, but package not picked up
    obs, reward, terminated, _, info = env.step(Action.UP)
    assert terminated is False
    assert info["stage"] == 0
    env.close()


def test_ansi_render_output():
    env = CourierRouteEnv(render_mode="ansi")
    env.reset(seed=0)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "Courier Route" in rendered
    assert "@" in rendered
    assert "C" in rendered
    assert "D" in rendered
    assert "Stage: package not collected" in rendered
    env.close()
