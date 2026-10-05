"""Training, the lambda sweep, and the learning-curve plot for PA2.

    python myrunner.py            # sweep, CSV, plot, summary table, report PDF
    python myrunner.py --check    # self-tests (see checks.py)

Every run is seeded, so the same command regenerates the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv  # noqa: F401  (importing registers the environment id)
import report
from myagent import ACCUMULATING, REPLACING, SarsaLambdaAgent

ENV_ID = "cs272/CourierRoute-v0"
LAMBDA_VALUES = (0.0, 0.3, 0.6, 0.9, 1.0)
DEFAULT_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
HYPERPARAMS = {"gamma": 0.99, "alpha": 0.08, "eps": 0.10, "init_val": 1.0}

TARGET_RETURN = 9.0     # the "reached" threshold used in the table
SMOOTH_WINDOW = 100     # trailing moving-average window used for curves and table
ZOOM_WINDOW = 10        # finer window for the early-learning panel of the plot
ZOOM_EPISODES = 200
DIP_AFTER = 500         # stability is judged on the curve after this episode
CREDIT_STEPS = (1, 3, 5, 11)  # steps back shown in the credit table; 5 and 11 are the first decision to the pickup and delivery errors
SAMPLE_SEED = 123       # seed of the agent that produces the report's sample episode

# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta),
# validated for colour-vision-deficiency separation.
LAMBDA_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def make_env(render_mode: str | None = None) -> gym.Env:
    return gym.make(ENV_ID, render_mode=render_mode)


# ---------------------------------------------------------------- training

def train(lam: float, seed: int, episodes: int, trace: str = ACCUMULATING) -> np.ndarray:
    """Train one SARSA(lambda) agent and return its per-episode returns."""
    env = make_env()
    agent = SarsaLambdaAgent(env, lam=lam, trace=trace, total_epi=episodes, seed=seed, **HYPERPARAMS)
    returns = agent.learn()
    env.close()
    return np.asarray(returns, dtype=float)


def _train_job(job: tuple[float, int, int, str]) -> np.ndarray:
    return train(*job)


def run_sweep(episodes: int, seeds: tuple[int, ...], workers: int = 1, trace: str = ACCUMULATING,
              lambdas: tuple[float, ...] = LAMBDA_VALUES) -> dict[float, np.ndarray]:
    """Train every (lambda, seed) pair. Returns {lambda: array (n_seeds, episodes)}.

    Runs are independent and individually seeded, so parallelism does not
    change any result.
    """
    jobs = [(lam, seed, episodes, trace) for lam in lambdas for seed in seeds]
    if workers <= 1:
        results = [_train_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_train_job, jobs))
    n = len(seeds)
    return {lam: np.asarray(results[i * n:(i + 1) * n]) for i, lam in enumerate(lambdas)}


# ------------------------------------------------------------------ metrics

def rolling_mean(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Trailing moving average; the first window-1 points average what exists so far."""
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    n = np.arange(1, len(values) + 1)
    return (cumulative[n] - cumulative[np.maximum(n - window, 0)]) / np.minimum(n, window)


def episodes_to_target(curve: np.ndarray, target: float = TARGET_RETURN, window: int = SMOOTH_WINDOW) -> int | None:
    """First episode whose smoothed return reaches the target, or None."""
    reached = np.flatnonzero(rolling_mean(curve, window) >= target)
    return int(reached[0]) + 1 if len(reached) else None


@dataclass
class LambdaStats:
    lam: float
    n_seeds: int
    first_mean_curve: int | None    # episodes to target on the seed-averaged curve
    first_per_seed_mean: float      # mean/sd over the seeds that reached the target
    first_per_seed_sd: float
    seeds_reached: int
    final_mean: float               # mean return of the last 100 episodes, over seeds
    final_sd: float
    worst_dip: float                # lowest smoothed return after episode DIP_AFTER, any seed


def lambda_stats(lam: float, runs: np.ndarray) -> LambdaStats:
    per_seed = [episodes_to_target(run) for run in runs]
    reached = np.array([v for v in per_seed if v is not None], dtype=float)
    final = runs[:, -100:].mean(axis=1)
    late = [rolling_mean(run)[DIP_AFTER:] for run in runs]
    return LambdaStats(
        lam=lam,
        n_seeds=len(runs),
        first_mean_curve=episodes_to_target(runs.mean(axis=0)),
        first_per_seed_mean=float(reached.mean()) if len(reached) else float("nan"),
        first_per_seed_sd=float(reached.std()) if len(reached) else float("nan"),
        seeds_reached=len(reached),
        final_mean=float(final.mean()),
        final_sd=float(final.std()),
        worst_dip=float(min((s.min() for s in late if len(s)), default=float("nan"))),
    )


def credit_weight(lam: float, gamma: float, steps_back: int) -> float:
    """Share of a TD error that reaches the pair visited `steps_back` steps earlier."""
    return (gamma * lam) ** steps_back


def fmt_first(value: int | None) -> str:
    return "not reached" if value is None else str(value)


# ------------------------------------------------------------------ outputs

def write_csv(path: Path, curves: dict[float, np.ndarray], seeds: tuple[int, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "seed", "episode", "return"])
        for lam, runs in curves.items():
            for seed, run in zip(seeds, runs):
                for episode, value in enumerate(run, start=1):
                    writer.writerow([f"{lam:g}", seed, episode, f"{value:.2f}"])


def _band(ax: plt.Axes, x: np.ndarray, runs: np.ndarray, window: int, color: str, label: str) -> None:
    smoothed = np.asarray([rolling_mean(run, window) for run in runs])
    mean, sd = smoothed.mean(axis=0), smoothed.std(axis=0)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)


def make_plot(curves: dict[float, np.ndarray], path: Path) -> None:
    """Learning curves, one line per lambda, band = +/- 1 sd across seeds.

    Top: the whole run with the 100-episode window. Bottom: the first episodes
    with a 10-episode window, because every lambda has learned the route within
    a few hundred episodes and the full-run panel compresses that part.
    """
    n_seeds, episodes = next(iter(curves.values())).shape
    zoom = min(ZOOM_EPISODES, episodes)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8.6), constrained_layout=True)
    for (lam, runs), color in zip(curves.items(), LAMBDA_COLORS):
        _band(top, np.arange(1, episodes + 1), runs, SMOOTH_WINDOW, color, f"λ = {lam:g}")
        _band(bottom, np.arange(1, zoom + 1), runs[:, :zoom], ZOOM_WINDOW, color, f"λ = {lam:g}")

    for ax, title, window in ((top, "Full run", SMOOTH_WINDOW), (bottom, f"First {zoom} episodes", ZOOM_WINDOW)):
        ax.axhline(TARGET_RETURN, color="#52514e", linestyle=":", linewidth=1, label=f"target = {TARGET_RETURN:g}")
        ax.set_title(f"{title} ({window}-episode moving average)", loc="left", fontsize=11)
        ax.set_xlim(0, ax.get_lines()[0].get_xdata()[-1])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return per episode")
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.set_ylim(4, 11.5)
    for ax in (top, bottom):
        ax.legend(ncol=3, loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"Courier Route: SARSA(λ) learning curves (mean of {n_seeds} seeds, band = ±1 sd)", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _summary_row(label: str, s: LambdaStats) -> str:
    return (f"| {label} | {fmt_first(s.first_mean_curve)} | {s.first_per_seed_mean:.0f} ± {s.first_per_seed_sd:.0f} "
            f"({s.seeds_reached}/{s.n_seeds} seeds) | {s.final_mean:.3f} ± {s.final_sd:.3f} | {s.worst_dip:.2f} |")


def write_summary(path: Path, stats: list[LambdaStats], ablation: LambdaStats,
                  seeds: tuple[int, ...], episodes: int) -> None:
    header = [
        "| lambda | episodes to target (mean curve) | episodes to target (per seed, mean ± sd) | "
        f"mean final return (last 100) | worst {SMOOTH_WINDOW}-episode mean after episode {DIP_AFTER} |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines = [
        "# Lambda sweep summary", "",
        f"Target: {SMOOTH_WINDOW}-episode moving average of the return >= {TARGET_RETURN:g}. "
        f"{len(seeds)} seeds ({', '.join(map(str, seeds))}), {episodes} episodes each, "
        f"hyperparameters {HYPERPARAMS}, accumulating traces.", "",
        *header, *(_summary_row(f"{s.lam:g}", s) for s in stats), "",
        "Ablation, lambda = 1 with replacing traces:", "",
        *header, _summary_row("1 (replacing)", ablation),
    ]
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------ sample episode

@dataclass
class SampleEpisode:
    lam: float
    seed: int
    episodes_trained: int
    frames: list[str]           # ansi render before the first step, then after each step
    actions: list[int]
    rewards: list[float]
    total_return: float
    reached_destination: bool


def sample_episode(lam: float, episodes: int, seed: int = SAMPLE_SEED) -> SampleEpisode:
    """Train one agent, then record one greedy episode with the ansi renderer.

    The episode is generated once and replayed under the same reset seed, so the
    rendered frames are exactly the transitions whose rewards are reported.
    """
    env = make_env(render_mode="ansi")
    agent = SarsaLambdaAgent(env, lam=lam, total_epi=episodes, seed=seed, **HYPERPARAMS)
    agent.learn()
    reset_seed = seed + 1
    episode, reached = agent.best_run(reset_seed=reset_seed)

    env.reset(seed=reset_seed)
    frames = [env.unwrapped.render()]
    for _, action, reward in episode:
        _, replay_reward, *_ = env.step(action)
        assert abs(replay_reward - reward) < 1e-12, "replay diverged from the recorded episode"
        frames.append(env.unwrapped.render())
    env.close()
    return SampleEpisode(lam, seed, episodes, frames, [a for _, a, _ in episode], [r for *_, r in episode],
                         agent.calc_return(episode), reached)


# --------------------------------------------------------------------- main

def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="run the self-tests and exit")
    parser.add_argument("--episodes", type=int, default=5000, help="number of training episodes per run")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS), help="number of seeds per lambda (at least 1)")
    parser.add_argument("--seeds-list", type=str, default=None, help="comma-separated list of explicit seeds (e.g. '11,22,33')")
    parser.add_argument("--trace", type=str, default=ACCUMULATING, choices=[ACCUMULATING, REPLACING],
                        help="eligibility trace mechanism (default: accumulating)")
    parser.add_argument("--lambdas", type=str, default=None, help="comma-separated lambda values (e.g. '0.0,0.5,1.0')")
    parser.add_argument("--no-pdf", action="store_true", help="skip generating the PDF report")
    parser.add_argument("--no-plot", action="store_true", help="skip generating the lambda sweep PNG plot")
    parser.add_argument("--output", type=Path, default=Path("results"), help="output directory for artifacts")
    parser.add_argument("--workers", type=int, default=default_workers(), help="parallel training processes")
    parser.add_argument("--repo-url", default="https://github.com/Kushagrabainsla/cs272-pa2-gym", help="URL for report link")
    args = parser.parse_args()

    if args.check:
        import checks
        checks.run_all()
        return
    if args.episodes < SMOOTH_WINDOW:
        raise SystemExit(f"need --episodes >= {SMOOTH_WINDOW}")

    if args.seeds_list is not None:
        try:
            seeds = tuple(int(s.strip()) for s in args.seeds_list.split(",") if s.strip())
            if not seeds:
                raise ValueError("empty seeds list")
        except ValueError as err:
            raise SystemExit(f"invalid --seeds-list: {err}")
    else:
        if args.seeds < 1:
            raise SystemExit("need --seeds >= 1")
        seeds = DEFAULT_SEEDS[:args.seeds] if args.seeds <= len(DEFAULT_SEEDS) else tuple(11 * k for k in range(1, args.seeds + 1))

    lambdas = LAMBDA_VALUES
    if args.lambdas is not None:
        try:
            lambdas = tuple(float(x.strip()) for x in args.lambdas.split(",") if x.strip())
            if not lambdas:
                raise ValueError("empty lambda list")
        except ValueError as err:
            raise SystemExit(f"invalid --lambdas: {err}")

    args.output.mkdir(parents=True, exist_ok=True)
    curves = run_sweep(args.episodes, seeds, args.workers, trace=args.trace, lambdas=lambdas)
    ablation_runs = run_sweep(args.episodes, seeds, args.workers, trace=REPLACING, lambdas=(1.0,))[1.0]

    stats = [lambda_stats(lam, runs) for lam, runs in curves.items()]
    ablation = lambda_stats(1.0, ablation_runs)
    write_csv(args.output / "returns.csv", curves, seeds)
    if not args.no_plot:
        make_plot(curves, args.output / "lambda_sweep.png")
    write_summary(args.output / "summary.md", stats, ablation, seeds, args.episodes)

    if not args.no_pdf:
        reached = [s for s in stats if s.first_mean_curve is not None]
        best = min(reached, key=lambda s: s.first_mean_curve) if reached else stats[0]
        sample = sample_episode(best.lam, args.episodes)

        settings = {
            "hyperparams": HYPERPARAMS, "target": TARGET_RETURN, "window": SMOOTH_WINDOW, "zoom_window": ZOOM_WINDOW,
            "dip_after": DIP_AFTER, "credit_steps": CREDIT_STEPS,
            "credit": {lam: [credit_weight(lam, HYPERPARAMS["gamma"], k) for k in CREDIT_STEPS] for lam in lambdas},
        }
        report.write_report(args.output / "report.pdf", repo_url=args.repo_url, plot_png=args.output / "lambda_sweep.png",
                            stats=stats, ablation=ablation, sample=sample, seeds=seeds, episodes=args.episodes,
                            settings=settings)

    print(f"Wrote results to {args.output}")
    print((args.output / "summary.md").read_text())


if __name__ == "__main__":
    main()
