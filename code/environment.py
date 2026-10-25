"""
environment.py

Environment classes for the k-armed bandit testbed.

The environment exposes exactly two public methods to the agent:
    - n_actions()  -> int
    - step(action) -> (reward: float, is_optimal: bool)
The agent never sees the true action-value array q_star. This prevents
information leakage from the environment into the agent.
"""

from math import sin

import numpy as np


class BanditEnvironment:
    """Base class for a k-armed bandit testbed."""

    def __init__(self, k=10,seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = np.zeros(k)  # true action values (hidden from agent)
        self.t = 0

    def reset(self):
        """Reset environment to its initial state. Subclasses override to
        set the initial q_star distribution."""
        self.t = 0
        self.q_star = np.zeros(self.k)

    def n_actions(self):
        return self.k

    def _optimal_action(self):
        return int(np.argmax(self.q_star))

    def _update_dynamics(self):
        """Hook called once per step, before the reward is drawn, so
        subclasses can implement non-stationarity. Base class: no-op
        (stationary bandit)."""
        pass

    def step(self, action):
        """Take an action, return (reward, was_action_optimal)."""
        optimal_action = self._optimal_action()
        reward = self.rng.normal(loc=self.q_star[action], scale=0.1)
        self.t += 1
        self._update_dynamics()
        return reward, (action == optimal_action)


class StationaryBandit(BanditEnvironment):
    """Standard 10-armed testbed: q_star drawn once from N(0,1) and fixed."""

    def __init__(self, k=10, seed=None):
        super().__init__(k, seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=self.k)


class RandomWalkBandit(BanditEnvironment):
    """
    Exercise 2.5 environment.

    All q_star(a) start EQUAL (typically all 0), then each takes an
    independent random walk (normal-distributed increment with mean 0
    and std `walk_std`) at every time step. This makes the problem
    non-stationary.
    """

    def __init__(self, k=10, walk_std=0.01, seed=None):
        super().__init__(k, seed)
        self.walk_std = walk_std
        self.reset()

    def reset(self):
        self.t = 0
        self.q_star = np.zeros(self.k)  # all arms start identical

    def _update_dynamics(self):
        self.q_star += self.rng.normal(0.0, self.walk_std, size=self.k)


class FlipBandit(BanditEnvironment):
    """
    Custom non-stationarity type #1: "arm flipping".

    Starts as a standard stationary 10-armed testbed (q_star ~ N(0,1)).
    Every `flip_interval` steps, a randomly chosen pair of arms swaps
    their true action values instantaneously. This is an abrupt, discrete
    form of non-stationarity (as opposed to the gradual random walk),
    which tests how quickly an agent's value estimates can "unlearn"
    a previously-good arm and pick up a previously-bad one.
    """

    def __init__(self, k=10, flip_interval=200, seed=None):
        super().__init__(k, seed)
        self.flip_interval = flip_interval
        self.reset()

    def reset(self):
        self.t = 0
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=self.k)

    def _update_dynamics(self):
        if self.t % self.flip_interval == 0:
            i, j = self.rng.choice(self.k, size=2, replace=False)
            self.q_star[i], self.q_star[j] = self.q_star[j], self.q_star[i]


class SinusodialBandit(BanditEnvironment):
    """
    Custom non-stationary environment: sinusoidally modulated random walk.
    
    At each time step, every arm's true action value q_star is updated by
    a random increment whose magnitude is modulated by sin(t). The random
    increment is sampled independently for each arm from a zero-mean normal
    distribution with standard deviation `walk_std`.

    Unlike the standard random-walk Bandit from Exercise 2.5, the amount
    of change is not constant over time. The environment's rate of change
    oscillates according to sin(t), periodically becoming positive,
    negative, or close to zero.
    """

    def __init__(self, k=10, seed=None):
        super().__init__(k, seed)
        self.period = 1000
        self.reset()

    def reset(self):
        self.t = 0
        self.phase = self.rng.uniform(0, 2 * np.pi, size=self.k)
        self.q_star = np.zeros(self.k)  # all arms start identical

    def _update_dynamics(self):
        self.q_star = np.sin(2 * np.pi * self.t / self.period + self.phase)
