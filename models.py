"""
models.py: the world models that the gate will judge.

In PROWL-2, the world model is a big neural network that "imagines" what
happens next. Here, every world model is a modified copy of the gridworld.
That sounds like cheating, but it's the point: because we build each model
by hand, we KNOW exactly where it is right and where it is wrong. That is
what lets us score the gate's decisions as false alarms or real catches.

The models:
  correct        identical rules to the real world. Every alarm on it is false.
  missing_wall   forgets one wall, so it lets you walk through it.  (structural error)
  teleport       stepping on one cell sends you across the map.     (structural error)
  overconfident  thinks ice never slips (believes p = 0).           (calibration error)

Each model has the same interface as GridWorld (set_state, step, rng), so the
rollout() and oracle_rollouts() functions from oracle.py work on them unchanged.
"""

import numpy as np
from env import FLOOR, GridWorld


# ---------------------------------------------------------------------------
# 1. The buggy models. Each is a "subclass" of GridWorld: it inherits all of
#    GridWorld's code and overrides only the part that's wrong.
# ---------------------------------------------------------------------------
class MissingWallMode(GridWorld):
    """Thinks the wall at (6,4) is floor."""
    
    #Why (6,4): its the wall just below the ice gap.
    #Paths that slip downward on the ice bump into it often
    #(you saw this in demo.py), so the bug shows up in a meaningful
    #number of windows, but not in all of them.
    WALL_CELL = (6,4)

    def __init__(self, p, **kwargs):
        #super().__init__ runs GridWorld's own setup first (map, rng, state).
        #**kwargs passes along any extra settings (e.g. max_steps) unchanged.
        super().__init__(p=p, **kwargs)
        #GridWorld built its own fresh grid array for this object, so editing
        # it here cannot affect the real environment or any other model
        self.grid[self.WALL_CELL] = FLOOR


class TeleportModel(GridWorld):
    """Stepping onto (5,2) sends you to (1,6), on the other side of the map"""
    # Why (5,2): its one of the most visited cells in the data (look at the coverage map
    #from data.py), so the bug is tested often.
    FROM_CELL = (5,2)
    TO_CELL = (1,6)

    def step(self, action):
        # RUn the normal step first, including its random draws...
        obs, reward, terminated, truncated, info = super().step(action)
        # ...then apply the bug. BEcause GridWorld.step already used this
        # step's random numbers, the bug never shifts the random sequence:
        #common random numbers still line up with the real environment
        if self.pos == self.FROM_CELL:
            self.pos = self.TO_CELL
            obs = self.pos
        return obs, reward, terminated, truncated, info
    

def make_model(name, p, **kwargs):
    """
    Build a world model by name, for a real world with randomness p.

    A small "factory" function like this means the experiment code can loop
    over model names (strings) without knowing how each one is built.
    """
    if name == "correct":
        return GridWorld(p=p, **kwargs) #same rules, same p
    if name == "missing_wall":
        return MissingWallMode(p=p, **kwargs)
    if name =="teleport":
        return TeleportModel(p=p, **kwargs)
    if name == "overconfident":
         # Believes the world is deterministic, whatever the real p is.
        # At p = 0 this model is actually CORRECT, which is a useful sanity
        # check: a "bug" only counts as an error where it changes outcomes.
        return GridWorld(p=0.0, **kwargs)
    raise ValueError(f"Unknown model: {name}")

MODEL_NAMES = ("correct","missing_wall","teleport","overconfident")

# ---------------------------------------------------------------------------
# 2. Measuring how wrong a model is: total variation (TV) distance
# ---------------------------------------------------------------------------
def tv_distance(dist_a, dist_b):
    """
    Total variation distance between two outcome distributions, each a dict
    {outcome: probability} like the ones end_distribution() returns.

    TV = half the sum of |P_a(x) - P_b(x)| over every outcome x.
      0   -> the two distributions are identical
      1   -> they never produce the same outcome
    Intuition: TV is the largest possible gap between the two distributions'
    probabilities for any single event. If TV = 0.3, there's some event that
    one model thinks is 30 percentage points more likely than the other does.

    This compares DISTRIBUTIONS, not single outcomes. It is how we will decide,
    with the oracle, whether a model is truly wrong for a given window. The
    single-reference gate can't do this; it only sees one real outcome.
    """
    outcomes = set(dist_a) | set(dist_b) #every outcome either side ever produced
    return 0.5*sum(abs(dist_a.get(x, 0.0) - dist_b.get(x, 0.0)) for x in outcomes)


# ---------------------------------------------------------------------------
# Running this file directly shows each model against the oracle on one window
# that passes by both planted bugs.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import config
    from env import DOWN, LEFT, RIGHT, EnvState
    from oracle import end_distribution, oracle_rollouts

    #Window: start on the ice at (4,3), then
    #   DOWN to (5,3), RIGHT to (5,4), DOWN into the wall at (6,4)
    #   (the missing_wall model walks thru it), LEFT, LEFT, onto (5,2)
    #   (the teleport model jumps to (1,6)).
    start = EnvState(pos=(4,3), t=0, done=False)
    actions = [DOWN, RIGHT, DOWN, LEFT, LEFT]
    n = 2000 #many replays, so the TV numbers are precose

    for p in (0.0, 0.3):
        truth = end_distribution(
            oracle_rollouts(GridWorld(p=p), start, actions, n_seeds=n, base_seed=config.ORACLE_SEED_BASE)
        )
        print(f"\np = {p}: true most likely ending {next(iter(truth))} "
              f"(prob {next(iter(truth.values())):.2f})")
        for name in MODEL_NAMES:
            model = make_model(name, p)
            pred = end_distribution(
                oracle_rollouts(model, start, actions, n_seeds=n, base_seed=config.MODEL_SEED_BASE)
            )
            print(f"    {name:14s} TV from truth = {tv_distance(pred, truth):.2f}")