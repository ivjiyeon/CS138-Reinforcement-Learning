"""
Tabular Monte Carlo control (Sutton & Barto, Ch. 5), written to work with any
discrete episodic environment that exposes:

    env.n_states, env.n_actions
    env.valid_key    : int array (n_states,)  -> row index into valid_masks
    env.valid_masks  : bool array (n_keys, n_actions)  which actions are legal
    env.valid_lists  : list of int arrays (legal action indices per key)
    env.reset()      -> state
    env.step(a)      -> (state, reward, done, info)
    env.stats        : dict of per-episode counters (crashes, collisions, ...)

Algorithms (selected with `mode`):
    'on_first'  on-policy first-visit MC control, epsilon-soft policy   (Sec 5.4)
    'on_every'  on-policy every-visit MC control, epsilon-soft policy
    'off_wis'   off-policy MC control, weighted importance sampling     (Sec 5.7)
                target = greedy w.r.t. Q, behaviour = epsilon-greedy w.r.t. current Q
"""
import numpy as np


class MCAgent:
    MODES = ("on_first", "on_every", "off_wis")

    def __init__(self, env, mode="on_first", epsilon=0.1, gamma=1.0,
                 q_init=0.0, seed=0):
        assert mode in self.MODES
        self.env, self.mode, self.eps, self.gamma = env, mode, epsilon, gamma
        self.rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        # Illegal actions get -inf so argmax never selects them.
        # q_init = 0 is *optimistic* when all rewards are negative, which
        # gives free exploration in the racetrack.
        self.Q = np.full((nS, nA), -np.inf, dtype=np.float32)
        self.Q[env.valid_masks[env.valid_key]] = q_init
        # visit counts (on-policy) or cumulative IS weights C (off-policy)
        self.N = np.zeros((nS, nA), dtype=np.float32)
        self.vkey, self.vlists = env.valid_key, env.valid_lists

    # ------------------------------------------------------------------ acting
    def act(self, s):
        """epsilon-greedy action and its probability under that policy."""
        valid = self.vlists[self.vkey[s]]
        n = len(valid)
        greedy = int(np.argmax(self.Q[s]))
        if self.rng.random() < self.eps:
            a = int(valid[int(self.rng.random() * n)])
        else:
            a = greedy
        b = self.eps / n + ((1.0 - self.eps) if a == greedy else 0.0)
        return a, b

    # ---------------------------------------------------------------- learning
    def _update_on_policy(self, S, A, R):
        first = None
        if self.mode == "on_first":
            first = {}
            for t, sa in enumerate(zip(S, A)):
                first.setdefault(sa, t)
        G = 0.0
        for t in range(len(S) - 1, -1, -1):
            G = self.gamma * G + R[t]
            s, a = S[t], A[t]
            if first is not None and first[(s, a)] != t:
                continue
            self.N[s, a] += 1.0
            self.Q[s, a] += (G - self.Q[s, a]) / self.N[s, a]

    def _update_off_policy(self, S, A, R, B):
        G, W = 0.0, 1.0
        for t in range(len(S) - 1, -1, -1):
            G = self.gamma * G + R[t]
            s, a = S[t], A[t]
            self.N[s, a] += W                       # C(s,a)
            self.Q[s, a] += (W / self.N[s, a]) * (G - self.Q[s, a])
            if a != int(np.argmax(self.Q[s])):      # target policy would not take a
                break
            W /= B[t]                               # pi(a|s)=1, b(a|s)=B[t]

    def run_episode(self, max_steps=500, learn=True):
        env = self.env
        s = env.reset()
        S, A, R, B = [], [], [], []
        done = False
        for _ in range(max_steps):
            a, b = self.act(s)
            s2, r, done, _ = env.step(a)
            S.append(s); A.append(a); R.append(r); B.append(b)
            s = s2
            if done:
                break
        if learn:
            if self.mode == "off_wis":
                self._update_off_policy(S, A, R, B)
            else:
                self._update_on_policy(S, A, R)
        out = dict(length=len(S), ret=float(sum(R)), success=float(done))
        out.update(env.stats)
        return out

    def train(self, n_episodes, max_steps=500, eval_every=0, n_eval=20,
              verbose=0):
        """Returns dict of per-episode arrays (+ 'eval_*' arrays if eval_every)."""
        hist = {}
        ev = {"eval_ep": [], "eval_length": [], "eval_ret": [], "eval_success": []}
        for ep in range(n_episodes):
            info = self.run_episode(max_steps)
            for k, v in info.items():
                hist.setdefault(k, []).append(v)
            if eval_every and (ep + 1) % eval_every == 0:
                e = evaluate(self.env, self.Q, n_eval, max_steps)
                ev["eval_ep"].append(ep + 1)
                ev["eval_length"].append(e["length"])
                ev["eval_ret"].append(e["ret"])
                ev["eval_success"].append(e["success"])
            if verbose and (ep + 1) % verbose == 0:
                w = hist["length"][-verbose:]
                print(f"    ep {ep+1:6d}  mean len (last {verbose}) = {np.mean(w):7.1f}",
                      flush=True)
        out = {k: np.array(v) for k, v in hist.items()}
        if eval_every:
            out.update({k: np.array(v) for k, v in ev.items()})
        return out


def evaluate(env, Q, n_episodes=100, max_steps=500):
    """Run the greedy policy w.r.t. Q (environment noise stays on)."""
    L, Rt, Su, extra = [], [], [], {}
    for _ in range(n_episodes):
        s = env.reset()
        total, done, t = 0.0, False, 0
        while not done and t < max_steps:
            s, r, done, _ = env.step(int(np.argmax(Q[s])))
            total += r
            t += 1
        L.append(t); Rt.append(total); Su.append(float(done))
        for k, v in env.stats.items():
            extra.setdefault(k, []).append(v)
    out = dict(length=float(np.mean(L)), ret=float(np.mean(Rt)),
               success=float(np.mean(Su)))
    out.update({k: float(np.mean(v)) for k, v in extra.items()})
    return out


def rollout(env, Q, max_steps=200):
    """One greedy episode; returns env.trace (needs env.record = True)."""
    env.record = True
    s = env.reset()
    done, t = False, 0
    while not done and t < max_steps:
        s, _, done, _ = env.step(int(np.argmax(Q[s])))
        t += 1
    env.record = False
    return dict(trace=list(env.trace), opp_trace=list(env.opp_trace),
                steps=t, finished=done)
