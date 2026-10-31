"""
agent.py

Agent classes for the k-armed bandit testbed.

Design note on the agent/environment interface:
    An agent only ever calls env.n_actions() and env.step(action). It
    maintains its OWN estimates Q(a) and never has access to the
    environment's true q_star. select_action() and update() are the only
    two methods the experiment loop calls on an agent.
"""

import numpy as np


class Agent:
    """Base class for a bandit agent."""

    def __init__(self, k, initial_value=0.0, step_size=0.1, seed=None):
        self.k = k
        self.initial_value = initial_value
        self.step_size = step_size  # alpha; constant step size handles
        # non-stationarity better than sample-average (1/n) weighting,
        # since it exponentially discounts old (stale) rewards.
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.Q = np.full(self.k, float(self.initial_value))
        self.N = np.zeros(self.k, dtype=int)
        self.t = 0

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.step_size * (reward - self.Q[action])
        self.t += 1


class EpsilonGreedyAgent(Agent):
    """
    Epsilon-greedy action selection.

    epsilon can be:
      - a fixed float (constant exploration rate), or
      - "decay": epsilon_t = epsilon0 / (1 + decay_rate * t)
    """

    def __init__(self, k, epsilon=0.1, decay=False, epsilon0=1.0,
                 decay_rate=0.001, min_epsilon=0.01, **kwargs):
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon0 = epsilon0
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        super().__init__(k, **kwargs)

    def current_epsilon(self):
        if not self.decay:
            return self.epsilon
        eps = self.epsilon0 / (1.0 + self.decay_rate * self.t)
        return max(eps, self.min_epsilon)

    def select_action(self):
        eps = self.current_epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.k))
        # break ties randomly among the max-valued actions
        best = np.flatnonzero(self.Q == self.Q.max())
        return int(self.rng.choice(best))


class SoftmaxAgent(Agent):
    """
    Softmax (Gibbs/Boltzmann) action selection over action-value estimates:
        P(a) = exp(Q(a)/tau) / sum_b exp(Q(b)/tau)

    tau (temperature) can be:
      - a fixed float, or
      - "decay": tau_t = max(tau_min, tau0 / (1 + decay_rate * t))
    Higher tau -> more uniform/exploratory; tau -> 0 -> greedy.
    """

    def __init__(self, k, tau=0.2, decay=False, tau0=1.0,
                 decay_rate=0.001, min_tau=0.05, **kwargs):
        self.tau = tau
        self.decay = decay
        self.tau0 = tau0
        self.decay_rate = decay_rate
        self.min_tau = min_tau
        super().__init__(k, **kwargs)

    def current_tau(self):
        if not self.decay:
            return self.tau
        tau = self.tau0 / (1.0 + self.decay_rate * self.t)
        return max(tau, self.min_tau)

    def select_action(self):
        tau = self.current_tau()
        # subtract max for numerical stability
        prefs = self.Q / tau
        prefs -= prefs.max()
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / exp_prefs.sum()
        return int(self.rng.choice(self.k, p=probs))
