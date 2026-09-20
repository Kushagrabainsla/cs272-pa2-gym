"""Courier Route, a small stochastic tabular Gymnasium environment."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class CourierRouteEnv(gym.Env):
    """Navigate a courier through a seven by seven warehouse map."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    SIZE = 7
    N_CELLS = SIZE * SIZE
    N_STATES = N_CELLS * 2
    START = (0, 0)
    CHECKPOINT = (3, 3)
    DESTINATION = (6, 6)
    SLIP_PROBABILITY = 0.15
    DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
    ACTION_SYMBOLS = ("U", "R", "D", "L")

    def __init__(self, render_mode: str | None = None):
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.observation_space = spaces.Discrete(self.N_STATES)
        self.action_space = spaces.Discrete(len(self.DELTAS))
        self.render_mode = render_mode
        self.row, self.col, self.stage = 0, 0, 0
        self.last_action: int | None = None
        self.last_reward = 0.0

    @classmethod
    def encode(cls, row: int, col: int, stage: int) -> int:
        return stage * cls.N_CELLS + row * cls.SIZE + col

    def _get_obs(self) -> int:
        return self.encode(self.row, self.col, self.stage)

    def _get_info(self) -> dict[str, int | bool]:
        return {"row": self.row, "col": self.col, "stage": self.stage, "has_package": bool(self.stage)}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.row, self.col, self.stage = self.START[0], self.START[1], 0
        self.last_action = None
        self.last_reward = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        action = int(action)
        direction = action
        if self.np_random.random() < self.SLIP_PROBABILITY:
            direction = int(self.np_random.choice(((action - 1) % 4, (action + 1) % 4)))
        d_row, d_col = self.DELTAS[direction]
        self.row = int(np.clip(self.row + d_row, 0, self.SIZE - 1))
        self.col = int(np.clip(self.col + d_col, 0, self.SIZE - 1))
        self.last_action = action

        reward = -0.02
        if self.stage == 0 and (self.row, self.col) == self.CHECKPOINT:
            self.stage = 1
            reward += 1.0
        terminated = self.stage == 1 and (self.row, self.col) == self.DESTINATION
        if terminated:
            reward += 10.0
        self.last_reward = float(reward)
        return self._get_obs(), self.last_reward, terminated, False, self._get_info()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = [
            "Courier Route",
            "Legend: C=checkpoint, D=destination, S=start, .=open, @=courier",
            f"Stage: {'package collected' if self.stage else 'package not collected'}",
        ]
        for r in range(self.SIZE):
            row_chars = []
            for c in range(self.SIZE):
                if (r, c) == (self.row, self.col):
                    row_chars.append("@")
                elif (r, c) == self.CHECKPOINT and self.stage == 0:
                    row_chars.append("C")
                elif (r, c) == self.DESTINATION:
                    row_chars.append("D")
                elif (r, c) == self.START:
                    row_chars.append("S")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        lines.append(f"Last action: {self.ACTION_SYMBOLS[self.last_action] if self.last_action is not None else '-'}")
        lines.append(f"Last reward: {self.last_reward:+.2f}")
        return "\n".join(lines)

    def close(self):
        pass


try:
    register(id="cs272/CourierRoute-v0", entry_point="myenv:CourierRouteEnv", max_episode_steps=100)
except gym.error.RegistrationError:
    pass


MyEnv = CourierRouteEnv
