"""
data.py: collect the "recorded experience" the gate will audit.

In PROWL-2, the gate looks at a start state from real experience, the actions
that were really taken from it, and the ONE real continuation that followed.
This file produces exactly that, for each randomness level p:

    start state  ->  H recorded actions  ->  the one real path that happened

Run with:  python data.py      (writes files into datasets/)
"""
import os
from collections import deque
import numpy as np
import config
from env import ICE, MOVES, N_ACTIONS, EnvState, GridWorld


# ---------------------------------------------------------------------------
# 1. Shortest-path distances, so the behaviour policy knows which way is "toward G"
# ---------------------------------------------------------------------------
def distances_to_goal(env):
    """
    For every cell, the number of steps to G along the shortest path
    (ignoring slips). Walls and unreachable cells get -1.

    Method: breadth-first search (BFS) outward from G. BFS explores cells in
    order of distance, so the first time it reaches a cell is via a shortest
    path. We search FROM the goal because moves are symmetric (if you can step
    from A to B you can step from B to A), so one search gives every cell's
    distance at once.
    """
    dist = np.full(env.grid.shape, -1, dtype = np.int64) #-1 means "not reached yet"
    dist[env.goal_pos]=0
    queue = deque([env.goal_pos]) #deque: a list that's fast to pop from the front
    while queue:
        r, c = queue.popleft() #the nearest cell not yet expanded
        for dr, dc in MOVES:
            # look at its 4 neighbors
            nb = (r+int(dr), c+int(dc))
            if env.is_free(nb) and dist[nb]==-1:
                #one step further than current cell
                dist[nb]=dist[r,c]+1
                queue.append(nb)
    return dist

# ---------------------------------------------------------------------------
# 2. The behaviour policy: "epsilon-greedy" on the shortest path
# ---------------------------------------------------------------------------
def behaviour_action(env, dist, rng, epsilon):
    """
    With probability epsilon, a uniformly random action (exploration).
    Otherwise, an action that moves one step closer to G (ties broken at random).

    Why this policy: PROWL-2's start states come from an agent that is trying
    to do the task but still makes mistakes. A purely random walker would
    rarely cross the ice to the right half of the map; a perfect walker would
    always take the same route. Epsilon-greedy sits in between and covers both
    halves of the map, including plenty of ice.
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    
    r,c = env.pos
    best=[]
    for a, (dr, dc) in enumerate(MOVES):
        nb = (r + int(dr), c+ int(dc))
        #a "best" action leads to a free cell exactly one step nearer the goal.
        if env.is_free(nb) and dist[nb]==dist[r,c]-1:
            best.append(a)

    #rng.choice picks one of the best actions at random, so the policy doesn't
    #always prefer e.g. "right" over "down" when both are equally good
    return int(rng.choice(best))
    


# ---------------------------------------------------------------------------
# 3. Collect episodes and cut them into windows
# ---------------------------------------------------------------------------
def collect(p, seed, n_starts=config.N_STARTS, horizon=config.HORIZON,
            epsilon=config.EPSILON, max_steps=config.MAX_STEPS,
            pool_factor=config.POOL_FACTOR):
    """
    Run the behaviour policy in the real environment and return a dataset of
    n_starts windows. Each window is: start state, `horizon` actions, real path.
    """
    # Three separate random generators, one per job (see the seed plan in config.py):
    env = GridWorld(p=p, max_steps=max_steps, seed=seed)                     # the world's luck
    policy_rng = np.random.default_rng(seed + config.POLICY_SEED_OFFSET)     # the agent's choices
    pick_rng = np.random.default_rng(seed + config.PICK_SEED_OFFSET)         # which windows we keep
    # Why separate: if the policy and the world shared one generator, changing
    # p would change how many numbers the world draws, which would change the
    # policy's choices too. Separate generators keep each job's luck its own.

    dist = distances_to_goal(env)
    # Episodes start at a random free cell (not G itself) rather than always at S.
    # Why: always starting at S would give lots of data near S and very little
    # elsewhere. Random starts ("exploring starts") spread the data over the map.
    start_cells = [cell for cell in env.free_cells() if cell != env.goal_pos]

    windows = []  # each item: (start_pos, start_t, actions, real_path)
    while len(windows) < n_starts * pool_factor:
        # --- play one episode ---
        env.reset(start=start_cells[policy_rng.integers(len(start_cells))])
        positions, actions = [env.pos], []
        while not env.done:
            a = behaviour_action(env, dist, policy_rng, epsilon)
            pos, _, _, _, _ = env.step(a)
            actions.append(a)
            positions.append(pos)

        # --- cut the episode into windows of `horizon` steps ---
        # A window can start at any step t that still has `horizon` real actions
        # after it. Windows near the end of an episode (fewer than `horizon`
        # actions left) are skipped rather than padded with made-up actions,
        # because a model would then be judged on actions nobody really took.
        # (Cost: states very close to G are slightly under-sampled. That goes in
        # the limitations list.)
        for t in range(len(actions) - horizon + 1):
            windows.append((
                positions[t],                      # where the window starts
                t,                                 # step count at that moment
                actions[t:t + horizon],            # the H actions really taken
                positions[t:t + horizon + 1],      # the H+1 positions really visited
            ))

    # Neighbouring windows from the same episode overlap heavily (window t and
    # t+1 share 9 of 10 steps). Collecting 5x more than we need and picking at
    # random means the kept windows are much less likely to be near-copies.
    keep = pick_rng.choice(len(windows), size=n_starts, replace=False)
    picked = [windows[i] for i in keep]

    # Turn the list of tuples into NumPy arrays, one array per field.
    start_pos = np.array([w[0] for w in picked], dtype=np.int64)        # (N, 2)
    start_t = np.array([w[1] for w in picked], dtype=np.int64)          # (N,)
    acts = np.array([w[2] for w in picked], dtype=np.int64)             # (N, H)
    real_paths = np.array([w[3] for w in picked], dtype=np.int64)       # (N, H+1, 2)

    # Labels used later for the "practice shift" half of H1.
    # ice_steps: how many of the H steps began on ice (i.e. could have slipped).
    # Index the grid with every (row, col) along the path at once: path[:, :-1]
    # are the positions where each action was taken.
    on_ice = env.grid[real_paths[:, :-1, 0], real_paths[:, :-1, 1]] == ICE  # (N, H) True/False
    ice_steps = on_ice.sum(axis=1)                                          # (N,)
    start_on_ice = env.grid[start_pos[:, 0], start_pos[:, 1]] == ICE        # (N,)

    return {
        "start_pos": start_pos,
        "start_t": start_t,
        "actions": acts,
        "real_paths": real_paths,
        "ice_steps": ice_steps,
        "start_on_ice": start_on_ice,
        "p": np.float64(p),
        "seed": np.int64(seed),
        "horizon": np.int64(horizon),
    }


# ---------------------------------------------------------------------------
# 4. Saving and loading
# ---------------------------------------------------------------------------
def dataset_path(p, seed, data_dir=config.DATA_DIR):
    """File name like datasets/p0.3_seed0.npz, so each dataset is easy to find."""
    return os.path.join(data_dir, f"p{p}_seed{seed}.npz")


def save_dataset(data, path):
    # .npz is NumPy's format for several named arrays in one file. Better than
    # CSV here because our arrays have different shapes (N, N x H, N x (H+1) x 2),
    # and it loads back exactly, with no rounding or text parsing.
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create datasets/ if missing
    np.savez_compressed(path, **data)                 # **data passes each key as a named array


def load_dataset(path):
    with np.load(path) as f:          # 'with' closes the file when done
        return {k: f[k] for k in f.files}


def start_state(data, i):
    """Rebuild window i's start as an EnvState, ready for the oracle or a model."""
    r, c = data["start_pos"][i]
    return EnvState(pos=(int(r), int(c)), t=int(data["start_t"][i]), done=False)


# ---------------------------------------------------------------------------
# 5. Running this file collects every dataset and prints a summary.
# ---------------------------------------------------------------------------
def coverage_map(data, env):
    """Text map showing how many start states begin in each cell."""
    counts = np.zeros(env.grid.shape, dtype=np.int64)
    # np.add.at adds 1 at each (row, col), correctly handling repeated cells.
    np.add.at(counts, (data["start_pos"][:, 0], data["start_pos"][:, 1]), 1)
    lines = []
    for r in range(env.grid.shape[0]):
        row = ""
        for c in range(env.grid.shape[1]):
            row += "   #" if not env.is_free((r, c)) else f"{counts[r, c]:4d}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    for p in config.P_VALUES:
        for seed in config.DATA_SEEDS:
            data = collect(p, seed)
            path = dataset_path(p, seed)
            save_dataset(data, path)
            touches_ice = (data["ice_steps"] > 0).mean()
            print(f"{path}: {len(data['start_pos'])} windows | "
                  f"{touches_ice:.0%} touch ice | "
                  f"{data['start_on_ice'].mean():.0%} start on ice | "
                  f"mean ice steps {data['ice_steps'].mean():.1f}/{config.HORIZON}")

    print("\nWhere the start states are (p=0.3, seed 0):")
    print(coverage_map(load_dataset(dataset_path(0.3, 0)), GridWorld()))
