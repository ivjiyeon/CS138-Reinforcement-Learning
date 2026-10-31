"""
Part 3: Windy Prize Race (see gridworld.py).

Exp A  algorithm sensitivity: {first-visit, every-visit, off-policy WIS} x
       {epsilon 0.1, 0.3} x {optimistic init q0 = 0, 25}; metric = how many seeds
       actually solve the race, and the return of the greedy policy.
Exp B  environment versions: wind strength max_wind in {0,1,2,3}.
Exp C  layout robustness: 10 different random layouts (wind columns + prize cell).
Policy figure: learned greedy policy and sample runs.
Value iteration with the exact model provides the optimal return as a reference.

    python part3.py --exp A --episodes 5000 --seeds 8
    python part3.py --exp B ; python part3.py --exp C ; python part3.py --exp policy
"""
import argparse, os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gridworld import WindyPrizeRace, ARROWS
from mc import MCAgent, evaluate, rollout
from viz import plot_curves

ALGOS = {"on-policy first-visit": "on_first", "on-policy every-visit": "on_every",
         "off-policy weighted IS": "off_wis"}
SOLVED = 0.9          # a seed is "solved" if the greedy policy finishes >= 90% of eval episodes


def run_one(mode, eps, q_init, seed, episodes, cap, layout_seed=0, max_wind=2,
            eval_every=250, n_eval=30, keep=False):
    env = WindyPrizeRace(layout_seed=layout_seed, max_wind=max_wind, seed=seed)
    ag = MCAgent(env, mode, epsilon=eps, q_init=q_init, seed=seed)
    h = ag.train(episodes, cap, eval_every=eval_every, n_eval=n_eval)
    fin = evaluate(env, ag.Q, 200, cap)
    out = dict(train_ret=h["ret"], eval_ep=h["eval_ep"], eval_succ=h["eval_success"],
               eval_ret=h["eval_ret"], final_ret=fin["ret"], final_succ=fin["success"])
    if keep:
        out.update(Q=ag.Q.copy(), N=ag.N.copy(), env=env)
    return out


def summarize(runs, vstar):
    succ = np.array([r["final_succ"] for r in runs]); ret = np.array([r["final_ret"] for r in runs])
    solved = succ >= SOLVED
    return dict(n=len(runs), solved=int(solved.sum()),
                ret_solved=float(ret[solved].mean()) if solved.any() else None,
                ret_solved_sd=float(ret[solved].std()) if solved.any() else None,
                ret_all=float(ret.mean()), vstar=vstar)


# ------------------------------------------------------------------ drawing
def draw_world(ax, env, Q=None, N=None, title=""):
    H, W = env.H, env.W
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.set_aspect("equal")
    for r in range(H):
        for c in range(W):
            fc = "#cfe3f7" if c in env.wind_cols else "white"
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc=fc, ec="lightgray", lw=.5))
            if c in env.wind_cols:
                ax.text(c, r + 0.38, "↑", ha="center", va="center", fontsize=6, color="#7aa6d6")
    pr, pc = env.prize
    ax.text(pc, pr, "★", ha="center", va="center", fontsize=16, color="goldenrod", zorder=3)
    for (r, c), lab, col in ((env.start, "S", "green"), (env.goal, "G", "crimson")):
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="none", ec=col, lw=2))
        ax.text(c - 0.33, r - 0.33, lab, fontsize=8, color=col, fontweight="bold")
    if Q is not None:
        for r in range(H):
            for c in range(W):
                for has, off, col in ((0, -0.2, "tab:blue"), (1, 0.2, "tab:red")):
                    s = env._enc(r, c, bool(has))
                    if N[s].sum() > 0:
                        ax.text(c + off, r, ARROWS[int(np.argmax(Q[s]))], ha="center",
                                va="center", fontsize=10, color=col, zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def draw_run(ax, env, trace, title):
    draw_world(ax, env, title=title)
    xs, ys = [p[1] for p in trace], [p[0] for p in trace]
    ax.plot(xs, ys, "-o", color="tab:blue", ms=3, lw=1, zorder=5)
    ax.plot(xs[0], ys[0], "s", color="green", ms=6, zorder=6)


# -------------------------------------------------------------------- main
GRID_A = [(0.1, 0.0), (0.3, 0.0), (0.1, 25.0), (0.3, 25.0)]           # (epsilon, initial Q)
CFGS_B = [("on-policy FV ($\\varepsilon$=0.3, Q0=25)", "on_first", 0.3, 25.0),
          ("off-policy WIS ($\\varepsilon$=0.3, Q0=0)", "off_wis", 0.3, 0.0)]
WINDS_B = [0, 1, 2, 3]
CFG_C = ("on_first", 0.3, 25.0)


def cache_path(out, exp, cell):
    return f"{out}/cache_p3_{exp}_{cell}.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "policy"], default="A")
    ap.add_argument("--cells", type=int, nargs="*", default=None,
                    help="which cells to RUN (A: 0-3 = (eps,Q0) grid cells, B: 0-1 = algorithm, "
                         "C: 0-9 = layouts); default all. Plots use every cached cell.")
    ap.add_argument("--layout", type=int, default=0, help="layout seed used by Exp B")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.episodes, a.seeds = 500, 2
    os.makedirs(a.out, exist_ok=True)
    E, cap = a.episodes, a.max_steps
    env0 = WindyPrizeRace(layout_seed=0)
    print(env0.render()); print("layout 0: wind columns", env0.wind_cols, "prize", env0.prize)

    # ---------------------------------------------------------------- Exp A
    if a.exp == "A":
        vstar = env0.optimal()[0]
        print(f"optimal V*(start) = {vstar:.2f}")
        for i in (a.cells if a.cells is not None else range(len(GRID_A))):
            eps, q0 = GRID_A[i]
            res = {}
            for name, mode in ALGOS.items():
                runs = [run_one(mode, eps, q0, s, E, cap) for s in range(a.seeds)]
                sm = summarize(runs, vstar)
                res[name] = dict(summary=sm, succ=np.array([r["eval_succ"] for r in runs]),
                                 ev_x=runs[0]["eval_ep"])
                print(f"eps={eps} q0={q0:4.0f} {name:24s} solved {sm['solved']}/{sm['n']}  "
                      f"return(solved) {sm['ret_solved']}", flush=True)
            pickle.dump(res, open(cache_path(a.out, "A", i), "wb"))
        cells = [i for i in range(len(GRID_A)) if os.path.exists(cache_path(a.out, "A", i))]
        fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 3.8), sharey=True, squeeze=False)
        table = {}
        for ax, i in zip(axes[0], cells):
            res = pickle.load(open(cache_path(a.out, "A", i), "rb")); eps, q0 = GRID_A[i]
            plot_curves(ax, {n: r["succ"] for n, r in res.items()},
                        "fraction of eval episodes finished", window=1,
                        xs=np.array(next(iter(res.values()))["ev_x"]))
            for n, r in res.items():
                table[f"{n}|eps={eps}|q0={q0}"] = r["summary"]
            ax.set_title(f"$\\varepsilon$={eps}, initial Q={q0:g}", fontsize=10); ax.set_ylim(-0.05, 1.05)
        fig.tight_layout(); fig.savefig(f"{a.out}/p3_expA_algorithms.png", dpi=150)
        json.dump(table, open(f"{a.out}/p3_expA.json", "w"), indent=1)

    # ---------------------------------------------------------------- Exp B
    if a.exp == "B":
        for i in (a.cells if a.cells is not None else range(len(CFGS_B))):
            cname, mode, eps, q0 = CFGS_B[i]
            res = {}
            for mw in WINDS_B:
                vstar = WindyPrizeRace(layout_seed=a.layout, max_wind=mw).optimal()[0]
                runs = [run_one(mode, eps, q0, s, E, cap, layout_seed=a.layout, max_wind=mw)
                        for s in range(a.seeds)]
                sm = summarize(runs, vstar)
                res[mw] = dict(summary=sm, ret=np.array([r["eval_ret"] for r in runs]), ev_x=runs[0]["eval_ep"])
                print(f"{cname} max_wind={mw}: solved {sm['solved']}/{sm['n']} return(solved) "
                      f"{sm['ret_solved']} (+-{sm['ret_solved_sd']}) optimal {vstar:.2f}", flush=True)
            pickle.dump(res, open(cache_path(a.out, f"B{a.layout}", i), "wb"))
        cells = [i for i in range(len(CFGS_B)) if os.path.exists(cache_path(a.out, f"B{a.layout}", i))]
        fig, axes = plt.subplots(1, len(cells), figsize=(6.5 * len(cells), 4), sharey=True, squeeze=False)
        table = {}
        for ax, i in zip(axes[0], cells):
            res = pickle.load(open(cache_path(a.out, f"B{a.layout}", i), "rb"))
            plot_curves(ax, {f"max wind {mw} (opt {r['summary']['vstar']:.1f})": r["ret"] for mw, r in res.items()},
                        "greedy-policy return (30 eval episodes)", window=1,
                        xs=np.array(next(iter(res.values()))["ev_x"]))
            ax.set_title(CFGS_B[i][0], fontsize=10)
            for mw, r in res.items():
                table[f"{CFGS_B[i][0]}|max_wind={mw}"] = r["summary"]
        fig.tight_layout(); fig.savefig(f"{a.out}/p3_expB_wind_L{a.layout}.png", dpi=150)
        json.dump(table, open(f"{a.out}/p3_expB_L{a.layout}.json", "w"), indent=1)

    # ---------------------------------------------------------------- Exp C
    if a.exp == "C":
        mode, eps, q0 = CFG_C
        for ls in (a.cells if a.cells is not None else range(10)):
            e = WindyPrizeRace(layout_seed=ls); vstar = e.optimal()[0]
            runs = [run_one(mode, eps, q0, s, E, cap, layout_seed=ls) for s in range(a.seeds)]
            sm = summarize(runs, vstar); sm.update(layout=ls, wind=list(e.wind_cols), prize=list(e.prize))
            print(f"layout {ls} wind {e.wind_cols} prize {e.prize}: solved {sm['solved']}/{sm['n']} "
                  f"return(solved) {sm['ret_solved']} optimal {vstar:.2f}", flush=True)
            pickle.dump(sm, open(cache_path(a.out, "C", ls), "wb"))
        rows = [pickle.load(open(cache_path(a.out, "C", i), "rb")) for i in range(10)
                if os.path.exists(cache_path(a.out, "C", i))]
        json.dump(rows, open(f"{a.out}/p3_expC.json", "w"), indent=1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        x = np.array([r["layout"] for r in rows])
        ax[0].bar(x - 0.2, [r["vstar"] for r in rows], 0.4, label="optimal (value iteration)")
        ax[0].bar(x + 0.2, [r["ret_solved"] if r["ret_solved"] is not None else np.nan for r in rows],
                  0.4, label="MC greedy (solved seeds)")
        ax[0].set_xlabel("layout"); ax[0].set_ylabel("return from start"); ax[0].legend(fontsize=8)
        ax[1].bar(x, [r["solved"] / r["n"] for r in rows])
        ax[1].set_xlabel("layout"); ax[1].set_ylabel("fraction of seeds solved"); ax[1].set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(f"{a.out}/p3_expC_layouts.png", dpi=150)

    # ------------------------------------------------------------- policy fig
    if a.exp == "policy":
        r = None
        for seed in range(10):                       # first seed that solves the race
            r = run_one("on_first", 0.3, 25.0, seed, E, cap, keep=True)
            if r["final_succ"] >= SOLVED:
                break
        env, Q, N = r["env"], r["Q"], r["N"]
        print(f"using seed {seed}: greedy return {r['final_ret']:.2f}, optimal {env.optimal()[0]:.2f}")
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        draw_world(axes[0, 0], env, Q, N, "Learned greedy policy (blue: no prize, red: has prize)")
        for i, ax in enumerate(axes.flat[1:]):
            tr = rollout(env, Q, cap)
            draw_run(ax, env, tr["trace"], f"greedy run {i+1}: {tr['steps']} steps")
        fig.tight_layout(); fig.savefig(f"{a.out}/p3_policy.png", dpi=150)
    print("done.")


if __name__ == "__main__":
    main()