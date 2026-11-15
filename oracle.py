"""
oracle.py: the ground truth fir Project A

The oracle answers one question: "From this exact start state, with these exact
actions, what are ALL the things that could happen, and how likely is each?"
It works by brute force: restore the start state, replay the actions, and repeat
with many different seeds (50 by default, as in the proposal)

WHy this matters for H1: PROWL-2's gate only ever sees ONE real continuation.
The oracle sees the whole spread. with it we can label every gate decision:
    - if a model's prediction is something that really could happen, a gate alarm on
    it is a FALSE alarm;
    - if the model predicts something the oracle says is impossible (or gets
    the odds badly wrong), an alarm is a real CATCH.
Real-world world models (like Odyssey's) have no oracle, because you can't reset
reality. That's exactly why we test on a gridworld first.
"""

from collections import Counter
import numpy as np

def rollout(env, start_state, actions, seed):
    """
    Replay `actions` from `start_state` using randomness from `seed`.

    Returns an array of shape (len(actions)+1,2): the start position followed
    by the position after each action. One row per time step, (row, col) per row.
    """
    #put the environment exactly where the start state says.
    env.set_state(start_state)
    #fresh generator for this replay. Same seed -> same random numbers ->
    #same path (our save/restore test proved this). Different seed -> new luck.
    env.rng = np.random.default_rng(seed)

    path = [start_state.pos] #the start position counts as step0
    for a in actions:
        if env.done: #reached the goal or the step limit: stop early
            break
        pos, _, _, _, _ = env.step(a) #we only need the new position here
        path.append(pos)

    #padding: if the episode ended early (E.g. reached G at step 3 of 10),
    #repeat the last position until the path has full length.
    #why: every path then has the same shape, so a batch of paths stacks inot
    #one NumPy array, and comparing two paths step by step always works
    #this treats the goal as "absorbing" (once there, you stay there), which
    #is the standard convention in RL.
    while len(path)<len(actions)+1:
        path.append(path[-1])

    return np.array(path, dtype = np.int64)


def oracle_rollouts(env, start_state, actions, n_seeds=50, base_seed=10_000):
    """
    Replay the same start state and actions under `n_seeds` different seeds.

    Returns an array of shape (n_seeds, len(actions) + 1, 2): one path per seed.

    base_seed: the oracle uses seeds base_seed, base_seed+1, ... We start at
    10,000 so these never overlap with the small seeds (0, 1, 2, ...) we will use
    for the "real" data and for the world models. Otherwise one of the oracle's
    50 replays could secretly be an exact copy of the real continuation, which
    would slightly bias the comparison.
    """
    #remember what the caller's environment was doing, so we can put it back.
    #the oracle should never secretly change someone else's environment: that
    #kind of hidden side effect causes very confusing bugs later
    saved_state = env.get_state()
    saved_rng = env.rng #rollout() swaps in new generators but never touches this object

    #try/finally: the 'finally' block runs even if something inside 'try'
    #crashes, so the environment is always restored.
    try:
        paths = [rollout(env, start_state, actions, base_seed+i) for i in range(n_seeds)]
    finally:
        env.set_state(saved_state)
        env.rng=saved_rng
    
    #np.stack turns a list of (H+1, 2) arrays into one (n_seeds, H+1, 2) array.
    return np.stack(paths)

def end_distribution(paths):
    """
    Summarise a batch of paths by where they END.

    Returns a dict {(row, col): probability}, sorted from most to least likely.
    Example: {(3,4): 0.78, (4,5): 0.08, ...}
    """
    #paths[:,-1] picks the last row of every path, i.e. every ending position.
    #tuple(...) beacuse dict keys must be hashable, and NumPy arrays aren't.
    endings = [tuple(int(x) for x in end) for end in paths[:,-1]]
    counts = Counter(endings) #e.g. Counter({(3,4): 39, (4,5): 4, ...})
    n = len(endings)
    return {pos: c / n for pos, c in counts.most_common()}

# ---------------------------------------------------------------------------
# Running this file directly (python oracle.py) shows the oracle at work.
# The 'if __name__ == "__main__"' guard means this block does NOT run when
# another file does 'import oracle', only when you run oracle.py itself.
# ---------------------------------------------------------------------------
if __name__ =="__main__":
    from env import DOWN,LEFT, RIGHT, UP, EnvState, GridWorld

    # 1. One step UP from the middle of the ice. We can work out the true
    #    answer by hand and check the oracle agrees:
    #      intended cell (3, 4): no slip (1 - p) + slip that picks UP (p/4) = 1 - 0.75p
    #      each of the other 3 neighbours: p/4
    p = 0.3
    env = GridWorld(p=p)
    start = EnvState(pos=(4,4), t=0, done=False)
    paths = oracle_rollouts(env, start, [UP], n_seeds=4000)
    print(f"One step UP from the ice centre (4,4), p={p}, 4000 replays:")
    print(f"  expected: (3,4) -> {1 - 0.75 * p:.3f}, each other neighbour -> {p / 4:.3f}")
    for pos, prob in end_distribution(paths).items():
        print(f"    oracle:     {pos} -> {prob:.3f}")


    #2. The 6-action walk across the ice from the demo, at several p values.
    actions = [DOWN, RIGHT, RIGHT, RIGHT, DOWN, DOWN]
    start = EnvState(pos=(2,3), t=0, done=False)
    print("\nTrue ending distribution for the demo walk (50 replays each):")
    for p in [0.0, 0.1, 0.3, 0.5]:
        dist = end_distribution(oracle_rollouts(GridWorld(p=p), start, actions))
        top3 = ", ".join(f"{pos}: {prob:.2f}" for pos, prob in list(dist.items())[:3])
        print(f"    p={p:.1f}: {len(dist)} endings; top 3 -> {top3}")
