"""
Part 1: single-car racetrack with MC control.

Compares on-policy first-visit MC (several epsilons) with off-policy weighted-IS MC.
Outputs (in --out): learning curves, trajectory plots of the trained policy,
optional GIF, and Q_<track>.npy (used by Part 2 as the opponent's policy).

    python part1.py --track A --episodes 20000 --seeds 5
    python part1.py --quick          # tiny smoke test
"""
import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
from racetrack import RaceTrack, make_track
from mc import MCAgent, evaluate, rollout
from viz import plot_curves, draw_trajectory, animate

CONDITIONS = {
    "on-policy FV, eps=0.05": dict(mode="on_first", epsilon=0.05),
    "on-policy FV, eps=0.10": dict(mode="on_first", epsilon=0.10),
    "on-policy FV, eps=0.30": dict(mode="on_first", epsilon=0.30),
    "off-policy WIS (behav. eps=0.10)": dict(mode="off_wis", epsilon=0.10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="A", choices=["A", "B"])
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--out", default="results")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds = 500, 2
    os.makedirs(args.out, exist_ok=True)

    layout = make_track(args.track)
    print("\n".join(layout))
    lengths, evals, ev_x, final_Q = {}, {}, None, {}
    eval_every = max(args.episodes // 20, 1)

    for name, kw in CONDITIONS.items():
        lengths[name], evals[name] = [], []
        for seed in range(args.seeds):
            t0 = time.time()
            env = RaceTrack(layout, seed=1000 + seed)
            agent = MCAgent(env, seed=seed, **kw)
            h = agent.train(args.episodes, args.max_steps, eval_every=eval_every, n_eval=20)
            lengths[name].append(h["length"]); evals[name].append(h["eval_length"])
            ev_x = h["eval_ep"]
            if seed == 0:
                final_Q[name] = agent.Q.copy()
            print(f"[{name}] seed {seed}: last-100 train len {h['length'][-100:].mean():.1f}, "
                  f"greedy eval len {h['eval_length'][-1]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        lengths[name] = np.array(lengths[name]); evals[name] = np.array(evals[name])

    # ---- learning curves (2 panels: behaviour-policy episodes, greedy evaluation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_curves(axes[0], lengths, "steps per training episode (smoothed)",
                window=max(args.episodes // 50, 1), logy=True)
    axes[0].set_title(f"Track {args.track}: training episodes (epsilon-greedy behaviour)")
    plot_curves(axes[1], evals, "steps, greedy policy (20 eval episodes)",
                window=1, logy=True, xs=np.array(ev_x))
    axes[1].set_title("Greedy-policy evaluation during training")
    fig.tight_layout()
    fig.savefig(f"{args.out}/p1_learning_curves_{args.track}.png", dpi=150)

    # ---- pick the best final policy (seed 0) and visualise it
    env = RaceTrack(layout, seed=7)
    scores = {n: evaluate(env, Q, 200, args.max_steps)["length"] for n, Q in final_Q.items()}
    print("\nGreedy eval (200 episodes, noise on):")
    for n, v in scores.items():
        print(f"   {n:35s} mean steps = {v:.1f}")
    best = min(scores, key=scores.get)
    Q = final_Q[best]
    np.save(f"{args.out}/Q_{args.track}.npy", Q)
    print(f"Best: {best} -> saved {args.out}/Q_{args.track}.npy")

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for i, (noise, ttl) in enumerate([(0.0, "no noise"), (0.1, "noise 0.1"), (0.1, "noise 0.1 (another run)")]):
        e = RaceTrack(layout, noise=noise, seed=50 + i)
        r = rollout(e, Q, 300)
        draw_trajectory(axes[i], e.grid, r["trace"], title=f"{ttl}: {r['steps']} steps")
        if args.gif and i == 0:
            animate(e.grid, r["trace"], None, f"{args.out}/p1_run_{args.track}.gif", title=ttl)
    fig.suptitle(f"Track {args.track}: trained policy ({best})")
    fig.tight_layout()
    fig.savefig(f"{args.out}/p1_trajectories_{args.track}.png", dpi=150)
    print("done.")


if __name__ == "__main__":
    main()
