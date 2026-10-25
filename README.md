# CS 138 Homework 1 — Multi-Armed Bandits in Non-Stationary Settings

Code README

## Contents

| File / Directory | Description |
|---|---|
| `environment.py` | Environment classes: `BanditEnvironment` (base class), `StationaryBandit`, `RandomWalkBandit` (Ex. 2.5), `FlipBandit` (Part 3 custom #1), `MeanRevertingShockBandit` (Part 3 custom #2). |
| `agent.py` | Agent classes: `Agent` (base class), `EpsilonGreedyAgent` (fixed or decaying epsilon), `SoftmaxAgent` (fixed or decaying temperature). |
| `run_experiments.py` | Runs Parts 1–3 and saves all plots to `./plots/`. |
| `plots/` | Output directory containing generated PNG plots. |

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
```

## How to Run

From this directory:

```bash
python3 run_experiments.py
```

This runs all three parts of the homework and writes PNG plots to `./plots/`:

| Output files | Experiment |
|---|---|
| `part1_reward.png`, `part1_optimal.png` | Part 1, Ex. 2.5 |
| `part2_reward.png`, `part2_optimal.png` | Part 2, epsilon-greedy vs. softmax |
| `part3_FlipBandit_reward.png`, `part3_FlipBandit_optimal.png` | Part 3, custom non-stationarity #1 |
| `part3_MeanRevertingShockBandit_reward.png`, `part3_MeanRevertingShockBandit_optimal.png` | Part 3, custom non-stationarity #2 |
| `part3_alpha_sweep_reward.png`, `part3_alpha_sweep_optimal.png` | Part 3, step-size hyperparameter sweep |

## Configuration

At the top of `run_experiments.py`:

```python
STEPS = 5000  # number of time steps per run
RUNS  = 150   # number of independent runs averaged together
```

Increase these for smoother, more publication-quality curves (at the cost of longer runtime); decrease them for faster iteration while developing. With `STEPS=5000` and `RUNS=150`, the full script runs in well under a minute on a standard laptop CPU.

## Agent/Environment Interface

Environments expose only:

```python
env.n_actions()  # -> int
env.step(action)  # -> (reward: float, was_optimal: bool)
```

Agents never see the environment's true action values (`q_star`); they only see the reward returned by `step()`. This is enforced by keeping `q_star` as an environment-internal attribute that agent code never reads.

Agents expose only:

```python
agent.select_action()  # -> int
agent.update(action, reward)
```

## Notes

- All environments use NumPy's Generator RNG (`np.random.default_rng`) with a per-run seed so that results are reproducible; environment and agent RNGs are seeded independently to avoid correlated randomness between the agent's exploration and the environment's dynamics.
- Ties in greedy action selection are broken uniformly at random.
- A **GEN-AI PROMPT LOG** must accompany this submission per the course policy; see `prompt_log.txt` (fill in / attach the conversation used to generate this code, as required by the syllabus).
