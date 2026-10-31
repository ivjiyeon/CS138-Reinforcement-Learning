# Sutton & Barto Ex. 5.12 – Racetrack (MC control)

Requires: numpy, matplotlib, pillow.

| file | contents |
|---|---|
| `mc.py` | tabular MC control: on-policy first/every-visit (eps-soft), off-policy weighted IS |
| `racetrack.py` | racetrack env (tracks A and B), optional second car, observation modes |
| `gridworld.py` | Part 3 Wind Prize World |
| `viz.py` | curves, trajectories, GIFs |
| `part1.py` / `part2.py` / `part3.py` | one script per assignment part |

## Run (order matters: Part 2 loads the policy Part 1 saves)
    python part1.py --track A --episodes 20000 --seeds 5 --gif   # ~5 min
    python part2.py --track A --episodes 20000 --seeds 3 --gif   # ~5 min
    python part3.py --exp A --episodes 5000 --seeds 8             # default experiment
    python part3.py --exp B --layout 0
    python part3.py --exp C
    python part3.py --exp policy
    # add --quick to any script for a smoke test; try --track B for the narrow track
Part 3 supports `--exp A`, `B`, `C`, or `policy` (default: `A`). `--cells` can
limit which cells/layouts are run: A uses 0-3 for the epsilon/Q0 combinations,
B uses 0-1 for the algorithms, and C uses 0-9 for layouts. Plots use all matching
cached cells, including results from earlier runs. For Part 3, `--quick` sets
episodes/seeds to 500/2; its defaults are 5000 episodes, 8 seeds, and a 200-step cap.

Outputs go to `results/` by default. Part 3 writes:
* Exp A: `p3_expA_algorithms.png`, `p3_expA.json`
* Exp B: `p3_expB_wind_L<layout>.png`, `p3_expB_L<layout>.json`
* Exp C: `p3_expC_layouts.png`, `p3_expC.json`
* Policy: `p3_policy.png`
* Cached experiment results: `cache_p3_*.pkl`

Exp A compares first-visit, every-visit, and off-policy weighted-IS MC across
epsilon values 0.1/0.3 and initial Q values 0/25. Exp B compares maximum wind
strengths 0-3 for two fixed configurations: on-policy first-visit
($\varepsilon$=0.3, Q0=25) and off-policy weighted IS
($\varepsilon$=0.3, Q0=0). Exp C evaluates first-visit MC
($\varepsilon$=0.3, Q0=25) on 10 layouts. The policy experiment plots a learned
policy and three sample greedy runs.

## Things to discuss in your write-up (observed while testing)
* Part 1: on-policy MC learns in ~1-3k episodes; off-policy WIS is far slower
  (only the tail of an episode after the last non-greedy action is used for updates).
* Part 2: compare `none` vs `pos` vs `posvel` observations and opponent offsets
  (opponent waits / starts together / head start). Watch collisions per episode as
  well as steps; a bigger state space also means slower learning.
* Part 3: the windy prize race ends only when the agent reaches the goal carrying
  the prize.The wind strength is sampled from 0 to `max_wind` each step. The experiment
  reports solved seeds (at least 90% success across final evaluation episodes) and
  greedy-policy returns. The value iteration supplies an optimal-return reference.
  Initial Q (`Q0`) is the starting action-value estimate for every valid action,
  before MC updates it. Policy arrows are shown only for visited states and are
  unreliable for rarely visited states.
