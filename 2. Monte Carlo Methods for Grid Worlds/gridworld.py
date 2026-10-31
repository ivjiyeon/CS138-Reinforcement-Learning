"""
Windy Prize Race (Part 3).

A grid race from the bottom-left corner (START) to the bottom-right corner (GOAL).

* WIND: `n_wind` randomly chosen columns (never the start/goal column) blow
  UPWARD. Each step, after the agent's own move, if the agent is in a windy
  column it is carried up by k cells, k ~ Uniform{0,...,max_wind} (a fresh random
  draw every step). The agent is carried cell by cell and stops at the top edge.
* PRIZE: one prize sits in a randomly chosen cell. It is picked up automatically
  when the agent enters (or is carried through) that cell.
* The race only FINISHES when the agent enters the goal cell while holding the
  prize. Entering the goal without the prize does nothing; the race continues.
* Reward: -1 per step, +prize_reward on pickup, +goal_reward on finishing.

The windy columns and the prize cell are drawn once, when the environment is
created (`layout_seed`), and stay fixed for all episodes; the randomness inside an
episode comes from the wind strength. State = (row, col, has_prize).
"""
import numpy as np

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ARROWS = "↑↓←→"


class WindyPrizeRace:
    def __init__(self, H=7, W=10, n_wind=3, max_wind=2, layout_seed=0, seed=0,
                 goal_reward=20.0, prize_reward=5.0, step_cost=1.0):
        self.H, self.W, self.max_wind = H, W, max_wind
        self.goal_reward, self.prize_reward, self.step_cost = goal_reward, prize_reward, step_cost
        self.start, self.goal = (H - 1, 0), (H - 1, W - 1)
        lay = np.random.default_rng(layout_seed)               # layout randomness
        self.wind_cols = tuple(sorted(int(c) for c in
                                      lay.choice(np.arange(1, W - 1), n_wind, replace=False)))
        cells = [(r, c) for r in range(H) for c in range(W)
                 if (r, c) not in (self.start, self.goal)]
        self.prize = cells[int(lay.integers(len(cells)))]
        self.rng = np.random.default_rng(seed)                 # wind randomness
        self.n_actions, self.n_states = 4, H * W * 2
        self.valid_masks = np.ones((1, 4), dtype=bool)
        self.valid_lists = [np.arange(4)]
        self.valid_key = np.zeros(self.n_states, dtype=np.int16)
        self.record = False
        self.opp_trace = []
        self.reset()

    # -------------------------------------------------------------- mechanics
    def _move(self, r, c, has, a, k):
        """Deterministic outcome of action a when the wind (if any) blows k cells.
        Returns (r, c, has, reward, done, windy)."""
        dr, dc = MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W):
            nr, nc = r, c                                      # bump into the border
        path = [(nr, nc)]
        windy = nc in self.wind_cols
        if windy:
            for i in range(1, k + 1):
                if nr - i < 0:
                    break
                path.append((nr - i, nc))
        reward = -self.step_cost
        for (pr, pc) in path:
            if (pr, pc) == self.prize and not has:
                has = True
                reward += self.prize_reward
            if (pr, pc) == self.goal and has:
                return pr, pc, has, reward + self.goal_reward, True, windy
        pr, pc = path[-1]
        return pr, pc, has, reward, False, windy

    def _enc(self, r=None, c=None, has=None):
        r = self.r if r is None else r
        c = self.c if c is None else c
        has = self.has if has is None else has
        return (r * self.W + c) * 2 + int(has)

    def reset(self):
        self.r, self.c = self.start
        self.has = False
        self.stats = {"wind_steps": 0}
        self.trace = [(self.r, self.c)]
        return self._enc()

    def step(self, a):
        k = int(self.rng.integers(self.max_wind + 1))
        self.r, self.c, self.has, reward, done, windy = self._move(self.r, self.c, self.has, a, k)
        self.stats["wind_steps"] += int(windy)
        if self.record:
            self.trace.append((self.r, self.c))
        return self._enc(), reward, done, {}

    # --------------------------------------------------- exact model (for VI)
    def outcome_tables(self):
        """T, R, D arrays of shape (nS, 4, K): next state, reward, done for each wind
        draw k = 0..max_wind. All K draws are equally likely."""
        K = self.max_wind + 1
        T = np.zeros((self.n_states, 4, K), dtype=int)
        R = np.zeros((self.n_states, 4, K))
        D = np.zeros((self.n_states, 4, K), dtype=bool)
        for r in range(self.H):
            for c in range(self.W):
                for has in (False, True):
                    s = self._enc(r, c, has)
                    for a in range(4):
                        for k in range(K):
                            r2, c2, h2, rew, d, _ = self._move(r, c, has, a, k)
                            T[s, a, k], R[s, a, k], D[s, a, k] = self._enc(r2, c2, h2), rew, d
        return T, R, D

    def optimal(self, iters=3000):
        """Value iteration with the exact model. Returns (V*(start), Q*)."""
        T, R, D = self.outcome_tables()
        V = np.zeros(self.n_states)
        for _ in range(iters):
            Vn = (R + np.where(D, 0.0, V[T])).mean(axis=2).max(1)
            done = np.max(np.abs(Vn - V)) < 1e-9
            V = Vn
            if done:
                break
        Q = (R + np.where(D, 0.0, V[T])).mean(axis=2)
        return float(V[self._enc(*self.start, False)]), Q

    def render(self):
        g = [["." for _ in range(self.W)] for _ in range(self.H)]
        for r in range(self.H):
            for c in self.wind_cols:
                g[r][c] = "^"
        g[self.prize[0]][self.prize[1]] = "P"
        g[self.start[0]][self.start[1]] = "S"
        g[self.goal[0]][self.goal[1]] = "G"
        return "\n".join("".join(row) for row in g)