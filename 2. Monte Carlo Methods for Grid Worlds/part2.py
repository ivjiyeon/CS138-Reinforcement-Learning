"""
Part 2: multi-agent racetrack. The other car follows the greedy Part-1 policy
(loaded from Q_<track>.npy; trained quickly here if missing).

Conditions: what the agent observes about the other car (none / pos / posvel)
            x opponent start offset (opp starts together / later / earlier).

    python part2.py --track A --episodes 20000 --seeds 3
    python part2.py --quick
"""
import argparse, os, time, csv
import numpy as np
import matplotlib.pyplot as plt
from racetrack import RaceTrack, make_track, single_car_greedy_policy
from mc import MCAgent, evaluate, rollout
from viz import plot_curves, draw_trajectory, animate


def load_opponent(track, out, layout, quick):
    path = f"{out}/Q_{track}.npy"
    if not os.path.exists(path):
        print("No Part-1 Q found; training a quick opponent policy...")
        env = RaceTrack(layout, seed=0)
        ag = MCAgent(env, "on_first", epsilon=0.05, seed=0)
        ag.train(3000 if not quick else 500, 300)
        os.makedirs(out, exist_ok=True)
        np.save(path, ag.Q)
    return single_car_greedy_policy(np.load(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="A", choices=["A", "B"])
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--penalty", type=float, default=10.0, help="extra cost of a collision")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 6, -6])
    ap.add_argument("--modes", nargs="+", default=["none", "pos", "posvel"])
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds, args.offsets = 500, 1, [0]
    os.makedirs(args.out, exist_ok=True)

    layout = make_track(args.track)
    opp = load_opponent(args.track, args.out, layout, args.quick)

    def make_env(mode, off, seed):
        return RaceTrack(layout, seed=seed, opponent_policy=opp, obs_mode=mode,
                         opp_offset=off, collision_penalty=args.penalty)

    res, rows, agents = {}, [], {}
    for off in args.offsets:
        for mode in args.modes:
            key = (mode, off)
            res[key] = {"length": [], "collisions": []}
            ev_list = []
            for seed in range(args.seeds):
                t0 = time.time()
                env = make_env(mode, off, 2000 + seed)
                ag = MCAgent(env, "on_first", epsilon=args.epsilon, seed=seed)
                h = ag.train(args.episodes, args.max_steps)
                res[key]["length"].append(h["length"])
                res[key]["collisions"].append(h["collisions"])
                # evaluate greedy policy in a fresh env with the same condition
                ev = evaluate(make_env(mode, off, 9000 + seed), ag.Q, 300, args.max_steps)
                ev_list.append(ev)
                if seed == 0:
                    agents[key] = ag.Q.copy()
                print(f"[obs={mode:6s} offset={off:+d}] seed {seed}: eval steps {ev['length']:.1f}, "
                      f"collisions/ep {ev['collisions']:.3f}, success {ev['success']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                del ag, env
            rows.append(dict(obs=mode, offset=off,
                             steps=np.mean([e["length"] for e in ev_list]),
                             steps_sd=np.std([e["length"] for e in ev_list]),
                             collisions=np.mean([e["collisions"] for e in ev_list]),
                             crashes=np.mean([e["crashes"] for e in ev_list]),
                             success=np.mean([e["success"] for e in ev_list])))
            for k in res[key]:
                res[key][k] = np.array(res[key][k])

    with open(f"{args.out}/p2_summary_{args.track}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nobs     offset  steps(+-sd)    collisions/ep  crashes/ep  success")
    for r in rows:
        print(f"{r['obs']:7s} {r['offset']:+5d}  {r['steps']:6.1f}({r['steps_sd']:4.1f})   "
              f"{r['collisions']:8.3f}      {r['crashes']:7.3f}    {r['success']:.2f}")

    # ---- learning curves: one column per offset
    n = len(args.offsets)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 7), squeeze=False)
    win = max(args.episodes // 50, 1)
    for j, off in enumerate(args.offsets):
        plot_curves(axes[0, j], {m: res[(m, off)]["length"] for m in args.modes},
                    "steps / episode", window=win, logy=True)
        axes[0, j].set_title(f"opponent offset {off:+d}")
        plot_curves(axes[1, j], {m: res[(m, off)]["collisions"] for m in args.modes},
                    "collisions / episode", window=win)
    fig.tight_layout()
    fig.savefig(f"{args.out}/p2_learning_curves_{args.track}.png", dpi=150)

    # ---- example runs of trained agents (first offset, each obs mode)
    off = args.offsets[0]
    fig, axes = plt.subplots(1, len(args.modes), figsize=(4.5 * len(args.modes), 6), squeeze=False)
    for i, m in enumerate(args.modes):
        env = make_env(m, off, 77)
        r = rollout(env, agents[(m, off)], args.max_steps)
        draw_trajectory(axes[0, i], env.grid, r["trace"], r["opp_trace"],
                        title=f"obs={m}, offset={off:+d}: {r['steps']} steps, "
                              f"{env.stats['collisions']} collisions")
        if args.gif:
            animate(env.grid, r["trace"], r["opp_trace"],
                    f"{args.out}/p2_run_{m}_{args.track}.gif", title=f"obs={m}")
    fig.tight_layout()
    fig.savefig(f"{args.out}/p2_trajectories_{args.track}.png", dpi=150)
    print("done.")


if __name__ == "__main__":
    main()
