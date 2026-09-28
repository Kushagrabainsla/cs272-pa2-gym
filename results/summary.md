# Lambda sweep summary

Target: 100-episode moving average of the return >= 9. 10 seeds (11, 22, 33, 44, 55, 66, 77, 88, 99, 110), 5000 episodes each, hyperparameters {'gamma': 0.99, 'alpha': 0.08, 'eps': 0.1, 'init_val': 1.0}, accumulating traces.

| lambda | episodes to target (mean curve) | episodes to target (per seed, mean ± sd) | mean final return (last 100) | worst 100-episode mean after episode 500 |
|---:|---:|---:|---:|---:|
| 0 | 78 | 73 ± 30 (10/10 seeds) | 10.665 ± 0.010 | 10.54 |
| 0.3 | 36 | 34 ± 17 (10/10 seeds) | 10.650 ± 0.017 | 10.58 |
| 0.6 | 25 | 25 ± 13 (10/10 seeds) | 10.647 ± 0.018 | 10.52 |
| 0.9 | 31 | 33 ± 19 (10/10 seeds) | 10.611 ± 0.051 | 10.10 |
| 1 | 101 | 72 ± 45 (10/10 seeds) | 10.611 ± 0.072 | 5.82 |

Ablation, lambda = 1 with replacing traces:

| lambda | episodes to target (mean curve) | episodes to target (per seed, mean ± sd) | mean final return (last 100) | worst 100-episode mean after episode 500 |
|---:|---:|---:|---:|---:|
| 1 (replacing) | 101 | 74 ± 42 (10/10 seeds) | 10.621 ± 0.052 | 9.30 |
