"""Training, checks, lambda sweep, and report generation for PA2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import textwrap

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import myenv
from myagent import RandomAgent, SarsaLambdaAgent

LAMBDA_VALUES = (0.0, 0.3, 0.6, 0.9, 1.0)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
TARGET_RETURN = 9.0


def make_env(render_mode: str | None = None):
    return gym.make("cs272/CourierRoute-v0", render_mode=render_mode)


def rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) < window:
        return np.cumsum(values) / np.arange(1, len(values) + 1)
    result = np.empty(len(values))
    result[: window - 1] = np.cumsum(values[: window - 1]) / np.arange(1, window)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def run_sweep(episodes: int, seeds: tuple[int, ...]) -> dict[float, np.ndarray]:
    curves: dict[float, np.ndarray] = {}
    for lam in LAMBDA_VALUES:
        runs = []
        for seed in seeds:
            env = make_env()
            agent = SarsaLambdaAgent(
                env, gamma=0.99, alpha=0.08, eps=0.10, lam=lam,
                trace="accumulating", total_epi=episodes, init_val=1.0, seed=seed,
            )
            runs.append(np.asarray(agent.learn(), dtype=float))
            env.close()
        curves[lam] = np.asarray(runs)
    return curves


def first_reach(curve: np.ndarray, target: float = TARGET_RETURN, window: int = 100) -> int | None:
    smooth = rolling_mean(curve, window)
    reached = np.flatnonzero(smooth >= target)
    return None if len(reached) == 0 else int(reached[0] + 1)


def write_csv(path: Path, curves: dict[float, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "seed_run", "episode", "return"])
        for lam, runs in curves.items():
            for run_number, run in enumerate(runs):
                for episode, value in enumerate(run, start=1):
                    writer.writerow([lam, run_number, episode, value])


def make_plot(curves: dict[float, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for lam, runs in curves.items():
        smoothed = np.asarray([rolling_mean(run) for run in runs])
        x = np.arange(1, smoothed.shape[1] + 1)
        mean, low, high = smoothed.mean(axis=0), smoothed.min(axis=0), smoothed.max(axis=0)
        ax.plot(x, mean, label=f"λ={lam:g}")
        ax.fill_between(x, low, high, alpha=0.12)
    ax.axhline(TARGET_RETURN, color="black", linestyle=":", linewidth=1, label=f"target={TARGET_RETURN:g}")
    ax.set(title="Courier Route SARSA(λ)", xlabel="Episode", ylabel="Return, 100-episode moving average")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_rows(curves: dict[float, np.ndarray]) -> list[tuple[float, int | None, float]]:
    rows = []
    for lam, runs in curves.items():
        mean_curve = runs.mean(axis=0)
        rows.append((lam, first_reach(mean_curve), float(runs[:, -100:].mean())))
    return rows


def make_report(output: Path, curves: dict[float, np.ndarray], repo_url: str) -> None:
    plot_path = output / "lambda_sweep.png"
    make_plot(curves, plot_path)
    rows = summary_rows(curves)
    env = make_env(render_mode="ansi")
    env.reset(seed=123)
    initial_map = env.unwrapped.render()
    agent = SarsaLambdaAgent(env, gamma=0.99, alpha=0.08, eps=0.10, lam=0.9,
                             total_epi=len(next(iter(curves.values()))[0]), init_val=1.0, seed=123)
    agent.learn()
    episode, success = agent.best_run()
    sample_lines = []
    env.reset(seed=123)
    sample_lines.append(env.unwrapped.render())
    for state, action, reward in episode:
        env.step(action)
        sample_lines.append(env.unwrapped.render())
    sample_return = agent.calc_return(episode)
    checkpoint_index = next((i for i, (_, _, reward) in enumerate(episode, start=1) if reward > 0.5), len(episode) // 2)
    selected_renders = [sample_lines[0], sample_lines[min(checkpoint_index, len(sample_lines) - 1)], sample_lines[-1]]
    action_trace = "\n".join(textwrap.wrap(" ".join(f"{action}:{reward:+.2f}" for _, action, reward in episode), width=86))
    env.close()

    with PdfPages(output / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "CS272 PA2: Courier Route and SARSA(λ)", fontsize=20, weight="bold")
        fig.text(0.08, 0.90, "Repository", fontsize=11, weight="bold")
        fig.text(0.08, 0.875, repo_url, fontsize=9, color="blue", url=repo_url)
        fig.text(0.08, 0.82, "Environment summary", fontsize=14, weight="bold")
        fig.text(0.08, 0.79, "The courier starts at the depot, collects a package at the center checkpoint, and delivers it at the lower-right destination. Each movement has a 15% perpendicular slip probability. The task rewards reaching both milestones while penalizing long routes.", wrap=True, fontsize=10, va="top")
        fig.text(0.08, 0.68, "State diagram and initial ANSI rendering", fontsize=14, weight="bold")
        fig.text(0.08, 0.65, "Stage 0: state = row*7 + column. Reach C to enter Stage 1.\nStage 1: state = 49 + row*7 + column. Reach D to terminate.\n\n" + initial_map, family="monospace", fontsize=8, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for lam, runs in curves.items():
            smoothed = np.asarray([rolling_mean(run) for run in runs])
            x = np.arange(1, smoothed.shape[1] + 1)
            mean, low, high = smoothed.mean(axis=0), smoothed.min(axis=0), smoothed.max(axis=0)
            ax.plot(x, mean, label=f"λ={lam:g}")
            ax.fill_between(x, low, high, alpha=0.12)
        ax.axhline(TARGET_RETURN, color="black", linestyle=":", linewidth=1)
        ax.set(title="Courier Route SARSA(λ)", xlabel="Episode", ylabel="Return, 100-episode moving average")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "Lambda sweep results", fontsize=18, weight="bold")
        fig.text(0.08, 0.90, f"Target: smoothed mean return >= {TARGET_RETURN:g}. Each cell uses the mean over five seeded runs.", fontsize=10)
        table_data = [[f"{lam:g}", "not reached" if first is None else str(first), f"{final:.3f}"] for lam, first, final in rows]
        table_ax = fig.add_axes([0.08, 0.70, 0.84, 0.15])
        table_ax.axis("off")
        table = table_ax.table(cellText=table_data, colLabels=["λ", "First episode at target", "Mean final return"], loc="center", cellLoc="center", bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        explanation = ("Eligibility traces distribute a temporal-difference error backward over recently visited state-action pairs. In this environment, the checkpoint reward is about six moves from the start, and the delivery reward is about six more moves after the checkpoint. With λ=0, each reward mainly updates the immediately preceding action, so information travels backward slowly across many episodes. Larger λ values send useful reward information farther back in the same episode, which can speed learning. Very large λ can also propagate noisy slips and boundary moves farther, so the best value depends on the noise, step penalty, and learning rate.")
        fig.text(0.08, 0.61, "Why λ changes the learning curve", fontsize=14, weight="bold")
        fig.text(0.08, 0.58, explanation, fontsize=10, wrap=True, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.94, "Sample greedy episode", fontsize=18, weight="bold")
        fig.text(0.08, 0.90, f"Reached destination: {success}. Return: {sample_return:.3f}. Each item is action:reward.", fontsize=10)
        fig.text(0.08, 0.86, action_trace, family="monospace", fontsize=8, va="top")
        fig.text(0.08, 0.78, "ANSI renderer snapshots: start, checkpoint, and terminal delivery", fontsize=12, weight="bold")
        y = 0.75
        for render in selected_renders:
            fig.text(0.08, y, render, family="monospace", fontsize=7, va="top")
            y -= 0.23
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    (output / "summary.md").write_text(
        "# Lambda sweep summary\n\n"
        "| lambda | first target episode | mean final return |\n|---:|---:|---:|\n" +
        "\n".join(f"| {lam:g} | {'not reached' if first is None else first} | {final:.3f} |" for lam, first, final in rows) +
        "\n\nTarget: smoothed mean return >= " + str(TARGET_RETURN) + ".\n"
    )


def run_checks() -> None:
    from gymnasium.utils.env_checker import check_env
    env = make_env()
    check_env(env, skip_render_check=False)
    assert env.observation_space.n == 98 and env.action_space.n == 4

    actions = [1, 1, 2, 2, 1, 2, 1, 2]
    trajectories = []
    for _ in range(2):
        state, _ = env.reset(seed=2026)
        trajectory = [state]
        for action in actions:
            state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward, terminated, truncated))
            if terminated or truncated:
                break
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1], "same seed did not reproduce trajectory"

    outcomes = set()
    for seed in range(30):
        env.reset(seed=seed)
        outcomes.add(env.step(1)[0])
    assert len(outcomes) > 1, "environment did not demonstrate stochastic outcomes"

    # For lambda=0, a trace only changes the current pair before it decays.
    test_env = make_env()
    agent = SarsaLambdaAgent(test_env, lam=0.0, total_epi=3, seed=7)
    returns = agent.learn()
    assert len(returns) == 3 and agent.q.shape == (98, 4)
    test_env.close()

    random_env = make_env()
    baseline = RandomAgent(random_env, total_epi=200, seed=8).learn()
    random_env.close()
    trained_env = make_env()
    trained = SarsaLambdaAgent(trained_env, lam=0.9, total_epi=1000, seed=8)
    trained_returns = trained.learn()
    trained_env.close()
    assert np.mean(trained_returns[-100:]) > np.mean(baseline[-100:]), "trained agent did not beat baseline"
    env.close()
    print("All checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repo-url", default="https://github.com/REPLACE_WITH_YOUR_PUBLIC_REPOSITORY")
    args = parser.parse_args()
    if args.check:
        run_checks()
        return
    if args.episodes <= 0 or args.seeds <= 0:
        raise ValueError("episodes and seeds must be positive")
    seeds = DEFAULT_SEEDS[:args.seeds] if args.seeds <= len(DEFAULT_SEEDS) else tuple(range(11, 11 + args.seeds))
    args.output.mkdir(parents=True, exist_ok=True)
    curves = run_sweep(args.episodes, seeds)
    write_csv(args.output / "returns.csv", curves)
    make_report(args.output, curves, args.repo_url)
    print(f"Wrote results to {args.output}")
    print((args.output / "summary.md").read_text())


if __name__ == "__main__":
    main()
