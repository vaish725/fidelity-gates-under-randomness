"""
config.py: every experiment setting in one place.

Why a config file: when the write-up says "H = 10, 1,000 start states, 3 seeds",
a reader (or you, in January) should be able to find those numbers in one
spot, and change them in one spot. Scattering magic numbers across files is
how experiments quietly end up using different settings.
"""

#randomness levels to sweep, as in the proposal: p in {0, 0.1, 0.3, 0.5}
P_VALUES = (0.0, 0.1, 0.3, 0.5)

#Horizon H: how many steps each recorded continuation (And each model rollout)
#lasts. Longer H = more chances to diverge, so this is a knob worth sweeping later
#(e.g. H = 5, 10, 20) as a robustness check.
HORIZON = 10

#how many start states the gate audits per (p, seed) dataset
N_STARTS = 1000

#independent repeats of the whole data collection, for error bars 
#(the proposal says at least 3 seeds)
DATA_SEEDS = (0, 1, 2)

#Behaviour policy: with probability EPSILON take a random action, 
# otherwise step along the shortest path to the goal.
#0.2 gives data that is mostly purposeful but still explores,
#like an agent part-way through learning
EPSILON = 0.3

#Episodes are cut off after this many steps. 
# Kept high so cut-offs are rare.
MAX_STEPS=200

#we collect POOL_FACTOR times more windows than we need, then pick
#N_STARTS at random.
# Explained in data.py (it reduces overlap between picked windows.)
POOL_FACTOR = 5

# ---------------------------------------------------------------------------
# Seed plan. Every random process gets its own non-overlapping range, so no two
# parts of the experiment ever share luck by accident.
#   data seeds      0, 1, 2        -> the "real" environment while collecting
#   policy seeds    100 + data seed -> the behaviour policy's random choices
#   picking seeds   200 + data seed -> which windows get picked as start states
#   model seeds     1,000+          -> world-model rollouts (week 2)
#   oracle seeds    10,000+         -> oracle replays (already used in oracle.py)
# ---------------------------------------------------------------------------

POLICY_SEED_OFFSET = 100
PICK_SEED_OFFSET = 200
MODEL_SEED_BASE = 1_000
ORACLE_SEED_BASE = 10_000

import os

#project folder, so paths work no matter where you run from
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

#where datasets are saved
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")

