"""Unit tests for myrunner utility functions and statistical aggregations."""

from __future__ import annotations

import numpy as np
import pytest

from myrunner import credit_weight, episodes_to_target, lambda_stats, rolling_mean


def test_rolling_mean():
    # Simple sequence [1, 2, 3, 4, 5] with window 3
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rm = rolling_mean(arr, window=3)
    # Expected:
    # i=0: 1/1 = 1.0
    # i=1: (1+2)/2 = 1.5
    # i=2: (1+2+3)/3 = 2.0
    # i=3: (2+3+4)/3 = 3.0
    # i=4: (3+4+5)/3 = 4.0
    assert np.allclose(rm, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_episodes_to_target():
    curve = np.array([1.0, 2.0, 3.0, 8.0, 10.0, 10.0])
    # Target 9.0 with window 2
    # rolling means:
    # 1.0, (1+2)/2=1.5, (2+3)/2=2.5, (3+8)/2=5.5, (8+10)/2=9.0 (index 4 -> episode 5)
    first_ep = episodes_to_target(curve, target=9.0, window=2)
    assert first_ep == 5

    # Target never reached
    assert episodes_to_target(curve, target=20.0, window=2) is None


def test_credit_weight():
    # credit_weight(lam, gamma, steps_back) = (gamma * lam) ** steps_back
    assert credit_weight(0.0, 0.99, 1) == 0.0
    assert credit_weight(1.0, 1.0, 5) == 1.0
    assert np.isclose(credit_weight(0.5, 0.8, 2), (0.4) ** 2)


def test_lambda_stats():
    # Create fake runs for 3 seeds over 600 episodes
    runs = np.full((3, 600), 10.0)
    # Give seed 0 a dip at episode 550
    runs[0, 550] = 5.0

    stats = lambda_stats(0.5, runs)
    assert stats.lam == 0.5
    assert stats.n_seeds == 3
    assert stats.seeds_reached == 3
    assert np.isclose(stats.final_mean, 9.983333, atol=1e-4)
    assert stats.worst_dip < 10.0
