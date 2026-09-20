# PA2: Courier Route and SARSA(λ)

This project implements the CS272 PA2 requirements. The custom environment is
documented in [env.md](env.md), and its registered ID is
`cs272/CourierRoute-v0`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Check the environment and agent

```bash
python myrunner.py --check
```

This checks the Gymnasium API, seeded trajectories, stochastic transitions,
environment dimensions, SARSA(0) behavior, and a random baseline comparison.

## Run the required sweep

```bash
python myrunner.py --episodes 5000 --seeds 5 --output results
```

This creates a CSV file, a smoothed learning-curve plot, a summary table, a
sample ANSI episode, and a report PDF. The default seeds are 11, 22, 33, 44,
and 55. Use `--repo-url` to place the final public repository link in the PDF.

## Files

- `myenv.py`: custom stochastic Gymnasium environment
- `myagent.py`: tabular SARSA(λ) with eligibility traces
- `myrunner.py`: checks, training, λ sweep, plot, and report generation
- `env.md`: complete environment specification
