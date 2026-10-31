"""
run_experiments.py

Runs Parts 1-3 of the homework and saves all plots to ./plots/.

Usage:
    python3 run_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from environment import (
    RandomWalkBandit, FlipBandit, SinusodialBandit
)
from agent import EpsilonGreedyAgent, SoftmaxAgent

STEPS = 5000      # steps per run
RUNS = 150        # independent runs to average over (reduce for speed)
K = 10
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def run_single(env_factory, agent_factory, steps, seed):
    """Run one (env, agent) pair for `steps` steps. Returns reward and
    optimal-action arrays of shape (steps,)."""
    env = env_factory(seed)
    agent = agent_factory(seed + 100000)  # decorrelate agent RNG from env RNG
    rewards = np.zeros(steps)
    optimal = np.zeros(steps, dtype=bool)
    for step in range(steps):
        action = agent.select_action()
        reward, is_opt = env.step(action)
        agent.update(action, reward)
        rewards[step] = reward
        optimal[step] = is_opt
    return rewards, optimal


def average_over_runs(env_factory, agent_factory, steps, runs):
    all_rewards = np.zeros((runs, steps))
    all_optimal = np.zeros((runs, steps))
    for run in range(runs):
        rewards, optimal = run_single(env_factory, agent_factory, steps, seed=run)
        all_rewards[run] = rewards
        all_optimal[run] = optimal
    return all_rewards.mean(axis=0), all_optimal.mean(axis=0) * 100.0


def plot_curves(curves, ylabel, title, fname, xlabel="Steps"):
    plt.figure(figsize=(8, 5))
    for label, y in curves.items():
        plt.plot(y, label=label, linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")

def environment_parameters():
    env_factory = lambda seed: RandomWalkBandit(k=K, walk_std=0.01, seed=seed)
    
    configs = {
        "constant-step eps-greedy (eps=0.1, step_size=0.1, initial_value=0.0)": lambda seed: EpsilonGreedyAgent(
            K, epsilon=0.1, step_size=0.1, seed=seed),
    }

    reward_curves, optimal_curves = {}, {}
    for label, agent_factory in configs.items():
        rewards, optimal = average_over_runs(env_factory, agent_factory, STEPS, RUNS)
        reward_curves[label] = rewards
        optimal_curves[label] = optimal

    plot_curves(reward_curves, "Average reward",
                "Average reward, random-walk bandit",
                "part0_reward_0_0.png")
# ---------------------------------------------------------------------
# PART 1: Exercise 2.5 - random walk non-stationarity
# ---------------------------------------------------------------------
def part1():
    env_factory = lambda seed: RandomWalkBandit(k=K, walk_std=0.01, seed=seed)
    
    configs = {
        "sample-avg eps-greedy (eps=0.1)": lambda seed: EpsilonGreedyAgent(
            K, epsilon=0.1, step_size=None, seed=seed),
        "constant-step eps-greedy (eps=0.1, step_size=0.1)": lambda seed: EpsilonGreedyAgent(
            K, epsilon=0.1, step_size=0.1, seed=seed),
    }

    # sample-average variant: override update() step size dynamically.
    class SampleAverageEpsGreedy(EpsilonGreedyAgent):
        def update(self, action, reward):
            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
            self.t += 1

    configs["sample-avg eps-greedy (eps=0.1)"] = lambda seed: SampleAverageEpsGreedy(
        K, epsilon=0.1, seed=seed)

    reward_curves, optimal_curves = {}, {}
    for label, agent_factory in configs.items():
        rewards, optimal = average_over_runs(env_factory, agent_factory, STEPS, RUNS)
        reward_curves[label] = rewards
        optimal_curves[label] = optimal

    plot_curves(reward_curves, "Average reward",
                "Part 1 (Ex. 2.5): Average reward, random-walk bandit",
                "part1_reward.png")
    plot_curves(optimal_curves, "% Optimal action",
                "Part 1 (Ex. 2.5): % optimal action, random-walk bandit",
                "part1_optimal.png")


# ---------------------------------------------------------------------
# PART 2: softmax vs. decaying-epsilon-greedy
# ---------------------------------------------------------------------
def part2():
    env_factory = lambda seed: RandomWalkBandit(k=K, walk_std=0.01, seed=seed)

    configs = {
        # "eps-greedy, fixed eps=0.1": lambda seed: EpsilonGreedyAgent(
        #     K, epsilon=0.1, step_size=0.1, seed=seed),
        "eps-greedy, decaying eps": lambda seed: EpsilonGreedyAgent(
            K, decay=True, epsilon0=1.0, decay_rate=0.005, min_epsilon=0.01,
            step_size=0.1, seed=seed),
        # "softmax, fixed tau=0.2": lambda seed: SoftmaxAgent(
        #     K, tau=0.2, step_size=0.1, seed=seed),
        "softmax, decaying temp": lambda seed: SoftmaxAgent(
            K, decay=True, tau0=1.0, decay_rate=0.005, min_tau=0.01,
            step_size=0.1, seed=seed),
    }

    reward_curves, optimal_curves = {}, {}
    for label, agent_factory in configs.items():
        rewards, optimal = average_over_runs(env_factory, agent_factory, STEPS, RUNS)
        reward_curves[label] = rewards
        optimal_curves[label] = optimal

    plot_curves(reward_curves, "Average reward",
                "Part 2: eps-greedy vs. softmax (random-walk bandit)",
                "part2_reward.png")
    plot_curves(optimal_curves, "% Optimal action",
                "Part 2: eps-greedy vs. softmax (random-walk bandit)",
                "part2_optimal.png")


# ---------------------------------------------------------------------
# PART 3: two custom non-stationarity types + hyperparameter comparison
# ---------------------------------------------------------------------
def part3():
    custom_envs = {
        "FlipBandit (flip_interval=1000)": lambda seed: FlipBandit(
            k=K, flip_interval=1000, seed=seed),
        "SinusodialBandit": lambda seed: SinusodialBandit(
            k=K, seed=seed),
    }

    agent_configs = {
        "eps-greedy eps=0.1": lambda seed: EpsilonGreedyAgent(
            K, epsilon=0.1, step_size=0.1, seed=seed),
        # "eps-greedy decaying": lambda seed: EpsilonGreedyAgent(
        #     K, decay=True, epsilon0=1.0, decay_rate=0.005, min_epsilon=0.01,
        #     step_size=0.1, seed=seed),
        "softmax tau=0.1": lambda seed: SoftmaxAgent(
            K, tau=0.1, step_size=0.1, seed=seed),
        # "softmax decaying tau": lambda seed: SoftmaxAgent(
        #     K, decay=True, tau0=1.0, decay_rate=0.005, min_tau=0.05,
        #     step_size=0.1, seed=seed),
    }

    for env_label, env_factory in custom_envs.items():
        reward_curves, optimal_curves = {}, {}
        for agent_label, agent_factory in agent_configs.items():
            rewards, optimal = average_over_runs(env_factory, agent_factory, STEPS, RUNS)
            reward_curves[agent_label] = rewards
            optimal_curves[agent_label] = optimal

        safe_name = env_label.split(" ")[0]
        plot_curves(reward_curves, "Average reward",
                    f"Part 3: Average reward, {env_label}",
                    f"part3_{safe_name}_reward.png")
        plot_curves(optimal_curves, "% Optimal action",
                    f"Part 3: % optimal action, {env_label}",
                    f"part3_{safe_name}_optimal.png")

    # step-size (alpha) hyperparameter sweep on the FlipBandit, using
    # decaying-epsilon-greedy, to illustrate the effect of alpha under
    # abrupt non-stationarity
    env_factory = custom_envs["FlipBandit (flip_interval=1000)"]
    alphas = [0.02, 0.1, 0.3, 0.5]
    reward_curves, optimal_curves = {}, {}
    for alpha in alphas:
        agent_factory = lambda seed, a=alpha: EpsilonGreedyAgent(
            K, decay=True, epsilon0=1.0, decay_rate=0.005, min_epsilon=0.01,
            step_size=a, seed=seed)
        rewards, optimal = average_over_runs(env_factory, agent_factory, STEPS, RUNS)
        reward_curves[f"step_size={alpha}"] = rewards
        optimal_curves[f"step_size={alpha}"] = optimal

    plot_curves(reward_curves, "Average reward",
                "Part 3: step-size sweep (FlipBandit, decaying eps-greedy)",
                "part3_alpha_sweep_reward.png")
    plot_curves(optimal_curves, "% Optimal action",
                "Part 3: step-size sweep (FlipBandit, decaying eps-greedy)",
                "part3_alpha_sweep_optimal.png")


if __name__ == "__main__":
    print(f"Running with STEPS={STEPS}, RUNS={RUNS} (edit these at the top of "
          f"this file to trade off speed vs. smoothness)")
    environment_parameters()
    print("Part 1...")
    part1()
    print("Part 2...")
    part2()
    print("Part 3...")
    part3()
    print("Done. Plots saved to ./plots/")
