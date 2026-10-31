"""
Racetrack environment (Sutton & Barto, Exercise 5.12) with an optional second car.

Conventions
-----------
* Grid cell (row, col); row 0 is the TOP of the picture. Velocity is (vr, vc):
  vr = cells moved UP per step, vc = cells moved RIGHT per step.
* Each velocity component is in [0, max_speed] and they cannot both be zero
  (except at the start line). Actions change each component by -1/0/+1
  (9 actions); illegal actions are masked out.
* Reward is -1 per step. Hitting a wall sends the car back to a random start
  cell with zero velocity (the episode continues). Crossing the finish line
  ends the episode.
* With probability `noise` (0.1) the velocity increments of a step are both 0.

Two-car version (Part 2)
------------------------
`opponent_policy` is a greedy policy (array: single-car state -> action) learned
in Part 1. The opponent obeys the same dynamics (including noise/crashes) and
respawns at the start after a crash; after finishing it leaves the track.
A collision happens when the cells swept by the two cars during one step
intersect. The agent is then sent back to the start (+ optional extra penalty).

`obs_mode` controls what the learning agent sees of the other car:
    'none'    nothing (agent is blind to the opponent)
    'pos'     relative position (d_row, d_col) if within `window` cells, else "far"
    'posvel'  as 'pos' plus a 1-bit "opponent is fast (speed >= 2)" flag
`opp_offset`: >0 opponent waits that many steps; <0 opponent gets a head start.
"""
import numpy as np

ACTIONS = [(dv_r, dv_c) for dv_r in (-1, 0, 1) for dv_c in (-1, 0, 1)]
WALL, TRACK, START, FINISH = 0, 1, 2, 3


# ----------------------------------------------------------------------- tracks
def make_track(name):
    """Return the track as a list of strings ('#' wall, '.' road, S start, F finish)."""
    if name == "A":      # 32 x 17, wide, gentle curves (similar to the book's track a)
        h, w = 32, 17
        rects = [(18, 31, 3, 8), (10, 17, 3, 11), (6, 9, 8, 12), (0, 5, 9, 15)]
        start_row, start_cols = 31, range(3, 9)
        fin_col, fin_rows = 16, range(0, 6)
    elif name == "B":    # 25 x 24, narrow (width 4) with a hairpin
        h, w = 25, 24
        rects = [(16, 24, 1, 4), (16, 19, 1, 14), (8, 19, 11, 14),
                 (8, 11, 3, 14), (0, 11, 3, 6), (0, 3, 3, 22)]
        start_row, start_cols = 24, range(1, 5)
        fin_col, fin_rows = 23, range(0, 4)
    else:
        raise ValueError(name)
    g = [["#"] * w for _ in range(h)]
    for r0, r1, c0, c1 in rects:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = "."
    for c in start_cols:
        g[start_row][c] = "S"
    for r in fin_rows:
        g[r][fin_col] = "F"
    return ["".join(row) for row in g]


def _path_table(V):
    tab = {}
    for vr in range(V):
        for vc in range(V):
            n = max(vr, vc)
            tab[(vr, vc)] = [(int(round(vr * i / n)), int(round(vc * i / n)))
                             for i in range(1, n + 1)] if n else []
    return tab


# ------------------------------------------------------------------ environment
class RaceTrack:
    def __init__(self, layout, noise=0.1, max_speed=5, seed=0,
                 opponent_policy=None, obs_mode="none", window=3,
                 opp_offset=0, collision_penalty=0.0):
        self.rng = np.random.default_rng(seed)
        self.layout = layout
        self.R, self.C = len(layout), len(layout[0])
        sym = {"#": WALL, ".": TRACK, "S": START, "F": FINISH}
        self.grid = np.array([[sym[ch] for ch in row] for row in layout], dtype=np.int8)
        self.starts = [(r, c) for r in range(self.R) for c in range(self.C)
                       if self.grid[r, c] == START]
        self.noise, self.V = noise, max_speed + 1
        self.n_actions = 9
        self.paths = _path_table(self.V)

        self.opp_policy = opponent_policy
        self.obs_mode, self.window = obs_mode, window
        self.opp_offset, self.collision_penalty = opp_offset, collision_penalty
        npos = (2 * window + 1) ** 2 + 1
        if opponent_policy is None or obs_mode == "none":
            self.K = 1
        elif obs_mode == "pos":
            self.K = npos
        elif obs_mode == "posvel":
            self.K = npos * 2
        else:
            raise ValueError(obs_mode)

        V = self.V
        self.n_states = self.R * self.C * V * V * self.K
        masks = np.zeros((V * V, 9), dtype=bool)
        for vr in range(V):
            for vc in range(V):
                for a, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = vr + dr, vc + dc
                    masks[vr * V + vc, a] = (0 <= nr < V and 0 <= nc < V
                                             and not (nr == 0 and nc == 0))
        self.valid_masks = masks
        self.valid_lists = [np.flatnonzero(m) for m in masks]
        self.valid_key = ((np.arange(self.n_states) // self.K) % (V * V)).astype(np.int16)
        self.record = False
        self.reset()

    # ------------------------------------------------------------- mechanics
    def _advance(self, r, c, vr, vc, a, noisy=True):
        """Apply action a to a car. Returns (status, r, c, vr, vc, swept_cells)."""
        dr, dc = ACTIONS[a]
        if noisy and self.rng.random() < self.noise:
            dr = dc = 0
        vr, vc = vr + dr, vc + dc
        cells = []
        for (pr, pc) in self.paths[(vr, vc)]:
            rr, cc = r - pr, c + pc
            if not (0 <= rr < self.R and 0 <= cc < self.C) or self.grid[rr, cc] == WALL:
                return "crash", r, c, vr, vc, cells
            if self.grid[rr, cc] == FINISH:
                return "finish", rr, cc, vr, vc, cells
            cells.append((rr, cc))
        if cells:
            r, c = cells[-1]
        return "ok", r, c, vr, vc, cells

    def _random_start(self, avoid=None):
        opts = [s for s in self.starts if s != avoid] or self.starts
        return opts[int(self.rng.integers(len(opts)))]

    def _opp_step(self):
        """Move the opponent one step; returns the cells it occupies/sweeps."""
        if self.opp_policy is None or self.opp_done:
            return []
        r, c, vr, vc = self.opp
        V = self.V
        s1 = ((r * self.C + c) * V + vr) * V + vc
        status, r, c, vr, vc, cells = self._advance(r, c, vr, vc, int(self.opp_policy[s1]))
        if status == "crash":
            r, c = self._random_start()
            vr = vc = 0
            cells = [(r, c)]
        elif status == "finish":
            self.opp_done = True
            self.opp = (r, c, vr, vc)
            return cells
        self.opp = (r, c, vr, vc)
        return cells if cells else [(r, c)]

    # ----------------------------------------------------------------- API
    def reset(self):
        self.r, self.c = self._random_start()
        self.vr = self.vc = 0
        self.t = 0
        self.stats = {"crashes": 0, "collisions": 0}
        self.trace, self.opp_trace = [(self.r, self.c)], []
        self.opp_done = self.opp_policy is None
        if self.opp_policy is not None:
            r, c = self._random_start(avoid=(self.r, self.c))
            self.opp = (r, c, 0, 0)
            for _ in range(max(0, -self.opp_offset)):      # head start
                self._opp_step()
            if self.record:
                self.opp_trace = [self.opp[:2] if not self.opp_done else None]
        return self._encode()

    def step(self, a):
        # opponent moves first (simultaneous move; opponent ignores the agent)
        opp_was_done = self.opp_done
        opp_cells = []
        if self.opp_policy is not None and self.t >= self.opp_offset:
            opp_cells = self._opp_step()
        elif self.opp_policy is not None and not self.opp_done:
            opp_cells = [self.opp[:2]]
        self.t += 1

        status, r, c, vr, vc, cells = self._advance(self.r, self.c, self.vr, self.vc, a)
        reward, done = -1.0, False
        if status == "finish":
            done = True
            self.r, self.c = r, c
        else:
            if status == "crash":
                self.stats["crashes"] += 1
                self.r, self.c = self._random_start()
                self.vr = self.vc = 0
            else:
                self.r, self.c, self.vr, self.vc = r, c, vr, vc
                if opp_cells and set(cells) & set(opp_cells):          # collision
                    self.stats["collisions"] += 1
                    reward -= self.collision_penalty
                    self.r, self.c = self._random_start()
                    self.vr = self.vc = 0
        if self.record:
            self.trace.append((self.r, self.c))
            self.opp_trace.append(
                self.opp[:2] if not self.opp_done or not opp_was_done else None
            )
        return self._encode(), reward, done, {}

    # ------------------------------------------------------------ observation
    def _obs_k(self):
        if self.K == 1 or self.opp_done:
            return 0
        w = self.window
        dr, dc = self.opp[0] - self.r, self.opp[1] - self.c
        if abs(dr) > w or abs(dc) > w:
            return 0
        k = 1 + (dr + w) * (2 * w + 1) + (dc + w)
        if self.obs_mode == "posvel":
            k = k * 2 + int(max(self.opp[2], self.opp[3]) >= 2)
        return k

    def _encode(self):
        V = self.V
        s = ((self.r * self.C + self.c) * V + self.vr) * V + self.vc
        return s * self.K + self._obs_k()


def single_car_greedy_policy(Q):
    """Greedy action table from a Part-1 Q table (illegal actions are -inf)."""
    return np.argmax(Q, axis=1).astype(np.int8)
