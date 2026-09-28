# Courier Route

| | |
|---|---|
| **Action space** | `Discrete(4)` |
| **Observation space** | `Discrete(98)` |
| **Environment id** | `cs272/CourierRoute-v0` |
| **Import** | `import myenv` (registers the id), then `gym.make("cs272/CourierRoute-v0")` |
| **Time limit** | 100 steps (`max_episode_steps` in `register()`) |

## Story

The agent is a courier on a 7 by 7 warehouse floor. Every shift starts at the
depot in the upper-left corner. The courier must first walk to the package
checkpoint in the middle of the floor and pick the package up, then carry it to
the delivery dock in the lower-right corner. The floor is slippery: about one
move in seven does not go where the courier steered. The courier is paid a small
penalty for every step, a bonus for picking up the package, and a large reward
for completing the delivery, so the best plan is the shortest route through
both stops. The checkpoint is on the way but is not the end: the shift only
ends when the package is delivered.

```
S . . . . . .        S = depot (start)      (0,0)
. . . . . . .        C = package checkpoint (3,3)
. . . . . . .        D = delivery dock      (6,6)
. . . C . . .        (row, column), both counted from 0, row 0 at the top
. . . . . . .
. . . . . . .
. . . . . . D
```

The two stops are each 6 moves from the previous one (the depot is 6 moves from
`C`, and `C` is 6 moves from `D`), so the large reward arrives about 12 steps
after the first decision that earns it.

## Action space

`Discrete(4)`: the direction the courier tries to move.

| Action | Direction | Change in `(row, column)` |
|---:|---|---|
| 0 | Up | `(-1, 0)` |
| 1 | Right | `(0, +1)` |
| 2 | Down | `(+1, 0)` |
| 3 | Left | `(0, -1)` |

Any other action raises `ValueError`. A move that would leave the floor is
clipped: the courier stays in its cell.

## Observation space

`Discrete(98)`: 2 stages times 49 cells. The stage records whether the package
has been picked up, so the same cell means different things before and after.

| Stage | Meaning |
|---:|---|
| 0 | package **not** collected |
| 1 | package collected |

The state is encoded into one integer exactly as

```text
state = stage * 49 + row * 7 + column          stage in {0,1}, row, column in 0..6
```

and decoded as `stage = state // 49`, `row = (state % 49) // 7`,
`column = state % 7`. Stage 0 uses ids 0 to 48 laid out like the floor (add 49
for stage 1):

```
  0   1   2   3   4   5   6
  7   8   9  10  11  12  13
 14  15  16  17  18  19  20
 21  22  23  24  25  26  27      <- 24 is the checkpoint C at (3,3)
 28  29  30  31  32  33  34
 35  36  37  38  39  40  41
 42  43  44  45  46  47  48      <- 48 is the dock D at (6,6), stage 0
```

Stage 1 delivery, `(6,6)` with the package, is state `49 + 48 = 97`; it is the
only terminal observation. Stage 0 at `(3,3)` (state 24) is never returned:
stepping onto the checkpoint immediately moves the courier to stage 1, so the
package pickup produces state 73. Stage 1 states 49 to 97 include the
checkpoint cell, which is then an ordinary cell.

`reset()` and `step()` also return an `info` dict holding the decoded state,
`{"row", "col", "stage", "has_package"}`; the agent does not need it.

## Starting state

Every `reset()` starts at `(row 0, column 0)` in stage 0, that is state `0`.
`reset(seed=k)` seeds the environment's `self.np_random`; the same seed followed
by the same actions gives exactly the same trajectory.

## Transition noise

This is the only randomness, and all of it is drawn from `self.np_random`. On
every step, before moving:

- with probability **0.85** the courier moves in the requested direction;
- with probability **0.15** it slips: with probability **0.075** to each of the
  two directions perpendicular to the requested one (for Up or Down these are
  Left and Right; for Left or Right they are Up and Down).

It never slips backwards or doubles its step. After the direction is chosen the
move is clipped at the walls, so a slip into a wall leaves the courier where it
was. Nothing else is random: the map, the rewards, and the start are fixed.

## Step order and rewards

One step, in this order:

1. Choose the actual direction (noise above) and move, clipping at the walls.
2. Reward `-0.02` for the step.
3. If the stage is 0 and the courier is now on `C` `(3,3)`: stage becomes 1 and
   the reward gains `+1.0`. This happens once per episode.
4. If the stage is 1 and the courier is now on `D` `(6,6)`: the reward gains
   `+10.0` and the episode terminates.

| Event | Reward |
|---|---:|
| every step (added to the rows below) | -0.02 |
| first arrival on the checkpoint | +1.0 |
| arrival on the dock after the pickup | +10.0 |

Walking over `D` before the pickup does nothing, and re-entering `C` after the
pickup pays nothing. Without slips the shortest route takes 12 steps and scores
`12 * -0.02 + 1 + 10 = 10.76`. A uniformly random agent scores about 0.3 on
average, because it rarely finds both stops inside 100 steps, and a trained
agent reaches about 10.6.

## Termination and truncation

- **Terminated** (`terminated=True`): the courier arrives at `(6,6)` in stage 1.
- **Truncated** (`truncated=True`): after 100 steps without a delivery. The
  environment itself never sets `truncated`; the `TimeLimit` wrapper added by
  `gym.make` does.

## Rendering

`render_mode="ansi"` makes `render()` return a string (with `render_mode=None`
it returns `None`). `@` is the courier and is drawn over anything, `C` is the
checkpoint until it has been collected, `D` is the dock, `S` is the depot when
the courier is elsewhere, and `.` is an open cell. Example, after the reset and
after six steps (`Down Down Right Right Right Down`, seed 0):

```
Courier Route
Legend: C=checkpoint, D=destination, S=start, .=open, @=courier
Stage: package not collected
@ . . . . . .
. . . . . . .
. . . . . . .
. . . C . . .
. . . . . . .
. . . . . . .
. . . . . . D
Last action: -
Last reward: +0.00
```

```
Courier Route
Legend: C=checkpoint, D=destination, S=start, .=open, @=courier
Stage: package not collected
S . . . . . .
. . . . . . .
. . @ . . . .
. . . C . . .
. . . . . . .
. . . . . . D
Last action: D
Last reward: -0.02
```

(The second frame shows the noise at work: without slips these six moves end on
`C` at `(3,3)`, but the courier is at `(2,2)`, two moves short, so the package
is still uncollected. `Last action` is the action that was requested, not the
direction actually taken.)

## Arguments and registration

```python
CourierRouteEnv(render_mode: str | None = None)
```

| Argument | Values | Meaning |
|---|---|---|
| `render_mode` | `None` (default), `"ansi"` | `"ansi"` enables text rendering; anything else raises `ValueError` |

There are no other arguments: the map size, the slip probability (0.15), and the
rewards are class constants in `myenv.py`.

```python
import gymnasium as gym
import myenv                                   # registers the id on import
env = gym.make("cs272/CourierRoute-v0", render_mode="ansi")
obs, info = env.reset(seed=0)
```

The id is registered with `entry_point="myenv:CourierRouteEnv"` and
`max_episode_steps=100`.

## Version history

- **v0**: first release.

---

# PA2: SARSA(λ) on Courier Route

CS272 PA2, build a world, then learn it. The environment above lives in
`myenv.py`, the agent in `myagent.py`, and the training, λ sweep and plot in
`myrunner.py`. The environment documentation is also kept as a standalone file,
[env.md](env.md) (an identical copy, checked by `python myrunner.py --check`).

## Files

| File | Purpose |
|---|---|
| `myenv.py` | the custom stochastic Gymnasium environment `cs272/CourierRoute-v0` |
| `myagent.py` | tabular SARSA(λ) with eligibility traces; knows nothing about the environment |
| `myrunner.py` | training, the λ sweep, the learning-curve plot, the summary table |
| `report.py` | lays out the submission PDF from the sweep results |
| `checks.py` | self-tests, run with `--check` |
| `env.md` | the environment documentation (same text as the top of this README) |
| `results/` | `report.pdf` (submitted), `lambda_sweep.png`, `summary.md`, `returns.csv` (every episode of every run) |

## Install and run

```bash
python -m pip install -r requirements.txt
python myrunner.py --check     # self-tests (a few seconds)
python myrunner.py             # full sweep (20 s to 2 min, by core count): CSV, plot, table, results/report.pdf
```

`python myrunner.py --help` lists the options (`--episodes`, `--seeds`,
`--output`, `--workers`, `--repo-url`).

## The agent

`SarsaLambdaAgent(env, gamma, alpha, eps, lam, trace, total_epi, init_val, seed)`
follows the pseudocode in the assignment: episodic traces, accumulating or
replacing, ties broken at random, `Q(terminal, .) = 0` never updated, and
truncation bootstraps from the real next state while termination does not. It
reads its table sizes off `env.observation_space.n` and `env.action_space.n`
and never imports `myenv`; `--check` also runs it on FrozenLake and Taxi.

## The λ sweep

λ ∈ {0, 0.3, 0.6, 0.9, 1.0}, 10 seeds each (11, 22, ..., 110), 5000 episodes,
γ = 0.99, α = 0.08, ε = 0.1, initial Q = 1, accumulating traces. The target is a
100-episode moving-average return of 9.0. Randomness is fully seeded: the agent
seeds its own generator, and once at the start of `learn()` it seeds the
environment from that generator (not with the same integer, which would give the
two generators identical streams). The same command therefore reproduces the
committed `results/`.

## Correctness checks

`python myrunner.py --check` verifies the Gymnasium API and the assignment's
environment requirements, seeded trajectories across fresh processes, and, for
the agent:

- λ = 0 gives exactly the Q table of an independent one-step SARSA;
- hand-computed trace updates on tiny scripted environments (λ propagation,
  accumulating vs replacing, truncation vs termination);
- the same seed gives the same learning curve;
- the trained agent clearly beats the uniform-random baseline.
