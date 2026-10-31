import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def smooth(x, w):
    """Trailing moving average along the last axis."""
    x = np.asarray(x, dtype=float)
    if w <= 1 or x.shape[-1] < w:
        return x
    k = np.ones(w) / w
    flat = x.reshape(-1, x.shape[-1])
    out = np.array([np.convolve(row, k, mode="valid") for row in flat])
    return out.reshape(x.shape[:-1] + (out.shape[-1],))


def plot_curves(ax, curves, ylabel, window=100, logy=False, xs=None):
    """curves: {label: array (n_seeds, n_points)}; mean +- std band over seeds."""
    for label, arr in curves.items():
        arr = np.atleast_2d(arr)
        sm = smooth(arr, window)
        x = (np.arange(sm.shape[-1]) + min(window, arr.shape[-1])) if xs is None \
            else xs[-sm.shape[-1]:]
        m, s = sm.mean(0), sm.std(0)
        ax.plot(x, m, label=label)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


_CMAP = ListedColormap(["#333333", "#eeeeee", "#9ad39a", "#f4a6a6"])  # wall, road, start, finish


def draw_track(ax, grid):
    ax.imshow(grid, cmap=_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def draw_trajectory(ax, grid, trace, opp_trace=None, title=""):
    draw_track(ax, grid)
    xs = [p[1] for p in trace]; ys = [p[0] for p in trace]
    ax.plot(xs, ys, "-o", color="tab:blue", ms=3, lw=1, label="agent")
    ax.plot(xs[0], ys[0], "s", color="green", ms=7)
    if opp_trace:
        pts = [(i, p) for i, p in enumerate(opp_trace) if p is not None]
        ax.plot([p[1] for _, p in pts], [p[0] for _, p in pts], "-o",
                color="tab:orange", ms=3, lw=1, alpha=0.8, label="other car")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")


def animate(grid, trace, opp_trace, path, title=""):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(4, 4 * grid.shape[0] / grid.shape[1] * 0.9))
    draw_track(ax, grid)
    (a_dot,) = ax.plot([], [], "o", color="tab:blue", ms=8)
    (a_tail,) = ax.plot([], [], "-", color="tab:blue", lw=1, alpha=0.5)
    (o_dot,) = ax.plot([], [], "o", color="tab:orange", ms=8)
    ttl = ax.set_title(title, fontsize=9)

    def upd(i):
        xs = [p[1] for p in trace[: i + 1]]; ys = [p[0] for p in trace[: i + 1]]
        a_dot.set_data([xs[-1]], [ys[-1]]); a_tail.set_data(xs, ys)
        if opp_trace and i < len(opp_trace) and opp_trace[i] is not None:
            o_dot.set_data([opp_trace[i][1]], [opp_trace[i][0]])
        else:
            o_dot.set_data([], [])
        ttl.set_text(f"{title} t={i}")
        return a_dot, a_tail, o_dot

    FuncAnimation(fig, upd, frames=len(trace), interval=200).save(
        path, writer=PillowWriter(fps=5))
    plt.close(fig)
