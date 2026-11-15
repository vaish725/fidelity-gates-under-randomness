"""
env.py: a small gridworld with tunable randomness, for Project A (testing H1).

This is the "real environment". Later, world models are judged against it, and the oravle
replays it many times from the same start state to learn the true spread of outcomes. 
3 properties matter for that:
    1. Randomness is controlled by one number, p, and only applies on ICE_CELL.
    2. The full state can be saved and restored exactly (needed by the oracle).
    3. All randomness comes from one seeded generator, so runs are reproducible.
"""

#lets us write type hints like tuple(int, int) in every Python 3.x version
from __future__ import annotations

#dataclass writes the boring __init__/__eq__/__repr__ for a small record type
from dataclasses import dataclass

#numpy holds the grid and provides the random number generator
import numpy as np

# ----------------------------------------------------------------------------
# 1. The map, written as text so anyone can read and edit it by eye.
# ----------------------------------------------------------------------------
#   #  wall    (cannot enter)
#   .  floor   (deterministic: you always move where you intended)
#   ~  ice     (random: with probability p your move goes in a random direction)
#   S  start   (a floor cell where episodes begin by default)
#   G  goal    (reaching it ends the episode with reward 1)
#
# Design choices:
# - Only the middle is icy, so the map has both deterministic states and random
#   states. That lets us test the second half of H1 ("practice moves away from
#   random states") by comparing how many ice states the gate keeps.
# - The wall down column 4 has its only gap through the ice, so any path from
#   S to G must cross the ice. Trajectories therefore visit random states often.
# - The interior wall segments also give us something to "forget" later, when
#   we build a buggy world model with a missing wall (a planted error).

DEFAULT_MAP = (
    "#########",
    "#S..#...#",
    "#...#...#",
    "#..~~~..#",
    "#..~~~..#",
    "#..~~~..#",
    "#...#...#",
    "#...#..G#",
    "#########",
)

#integer codes for each cell type. Integers in a NumPy array are fast
#look up and easy to compare, unlike a grid of strings
FLOOR, WALL, ICE, GOAL = 0,1,2,3

#how each map character translates into a cell code. "S" is just floor;
#its position is stored separately as the start
CHAR_TO_CELL = {".": FLOOR, "S": FLOOR, "#": WALL, "~": ICE, "G": GOAL}

# ----------------------------------------------------------------------------
# 2. Actions. Positions are (row, col); row grows downward, like a matrix.
# ----------------------------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0,1,2,3
N_ACTIONS = 4
ACTION_NAMES = ("up","right","down","left")

#MOVES[a] is the (row change, col change) for action a
#UP is row-1 cuase row 0 is the top of the map
MOVES = np.array([[-1,0],[0,1],[1,0],[0,-1]])

# ----------------------------------------------------------------------------
# 3. The saved state: everything needed to put the environment back exactly.
# ----------------------------------------------------------------------------
#frozen = True makes it read-only, so a saved state can't be changed by accident
#after you store it (eg: in a list of start states for the oracle).
@dataclass(frozen=True)
class EnvState:
    pos: tuple[int, int] #where the agent is
    t: int #how many steps have been taken this episode
    done: bool #whether the episode has ended

def parse_map(layout):
    """Turn the text map into an integer grid and find the S and G positions"""
    #every row must be the same width, or the grid isnt a rectangle
    width = len(layout[0])
    if any(len(row) != width for row in layout):
        raise ValueError("All map rows must have the same length")
    
    #start with an all-zero (all FLOOR) grid, then fill in each cell
    grid = np.zeros((len(layout), width), dtype=np.int8)
    start = goal = None

    #enumerate gives (index, item) pairs, so r is the row number and c the column
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            grid[r,c] = CHAR_TO_CELL[ch] #a KeyError here means an unknown character
            if ch =="S":
                start = (r,c)
            elif ch == "G":
                goal = (r,c)

    if start is None or goal is None:
        raise ValueError("The map needs exactly one S and one G")
    return grid, start, goal


class GridWorld:
    """
    A gridworld with "ice" cells where moves slip with probability p.

    The interface copies the Gymnasium convention (reset() and step() returning
    obs, reward, terminated, truncated, info) so it will feel familiar if you
    use standard RL libaries later, without needing to install them now.
    """
    def __init__(self, p=0.0, layout=DEFAULT_MAP, max_steps=100, seed=None):
        #p is a probability, so reject anything outside [0,1] right away
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p=p
        self.max_steps = max_steps #episode is cut off after this many steps
        self.grid, self.start_pos, self.goal_pos = parse_map(layout)

        #one private random generator for this environment.
        #default_rng is NumPy's modern generator;
        #giving it a seed makes runs repeatable
        self.rng = np.random.default_rng(seed)

        #the live state.
        #reset() fills these in properly
        self.pos = self.start_pos
        self.t = 0
        self.done = False

    # ---------------------------------------------------------------- helpers
    def in_bounds(self, pos):
        """True if pos is inside the grid's rectangle"""
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]
    
    def is_free(self, pos):
        """True if the agent can stand on pos (inside the grid and not a wall)"""
        return self.in_bounds(pos) and self.grid[pos]!=WALL
    
    def is_ice(self, pos):
        """True if pos is an ice cell (Where slipping can happen)"""
        return self.grid[pos]==ICE
    
    def free_cells(self):
        """Every cell the agent could stand on, as a list of (row, col)"""
        rows, cols = np.nonzero(self.grid != WALL)
        return [(int(r), int(c)) for r,c in zip(rows, cols)]
    
    # ------------------------------------------------------- episode control
    def reset(self, start=None, seed=None):
        """Start a new episode, at S or at a given start position"""
        #optionally re-seed, so an episode can be reproduced on its own
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        #tuple(...) accepts lists or arrays too;
        #int(...) avoids NumPy ints
        #sneaking into positions (they print and compare slightly differently)
        pos = self.start_pos if start is None else (int(start[0]), int(start[1]))
        if not self.is_free(pos):
            raise ValueError(f"Start {pos} is a wall or outside the grid.")
        self.pos, self.t, self.done = pos, 0, False
        #the observation is just the position: this world is fully observed
        return self.pos
    
    def get_state(self):
        """Snapshot of the current state (does NOT include the random generator)"""
        return EnvState(pos=self.pos, t=self.t, done=self.done)
    
    def set_state(self, state):
        """Put the environment back into a saved state"""
        if not self.is_free(state.pos):
            raise ValueError(f"cannot set state at {state.pos}: wall or out of bounds.")
        self.pos, self.t, self.done = state.pos, state.t, state.done

    #why the snapshot leaves out the random generator: the oracle's whole job
    #is to restore the SAME state and replay with DIFFERENT randomness (50 seeds),
    #so the state and the randomness need to be separate things

    # ------------------------------------------------------------- dynamics
    def step(self, action):
        """Take one action. Returns (obs, reward, terminated, truncated, info)"""
        if self.done:
            raise RuntimeError("Episode is over; call reset() first.")
        if action not in range(N_ACTIONS):
            raise ValueError(f"Action must be 0-3, got {action}")
        action = int(action)

        # Draw the randomness for this step EVERY time, even off the ice.
        # Why: this keeps the generator's sequence lined up step by step. Two
        # runs with the same seed then use the same random numbers at step k
        # no matter which cells they visited. Later, a buggy world model and the
        # real environment can share a seed so that any difference between them
        # comes from the bug alone, not from different luck. (This trick is
        # called "common random numbers".)
        u = self.rng.random() #uniform number in [0,1)
        random_dir = int(self.rng.integers(N_ACTIONS)) #0, 1, 2 or 3

        #slipping depends on the cell you are standing on BEFORE moving
        on_ice = bool(self.is_ice(self.pos))
        slipped = on_ice and u<self.p #happens with probability p, only on ice

        #a slip replaces the intended action with a uniformly random direction
        #that random direction can equal the intended one (Chance 1/4), so the
        # move actually goes somewhere unintended with probability p*3/4
        actual = random_dir if slipped else action

        #work out the target cell; moving into a wall means staying put
        dr, dc = MOVES[actual]
        target = (self.pos[0] + int(dr), self.pos[1] + int(dc))
        if self.is_free(target):
            self.pos = target
        
        self.t += 1
        terminated = self.pos == self.goal_pos #reached the goal
        truncated = (not terminated) and self.t>=self.max_steps #ran out of time
        self.done = terminated or truncated
        reward = 1.0 if terminated else 0.0 #sparse reward; only used later for agents (H2)

        #info is for debugging and tests, not for the agent. Recording the
        #slip lets us check that slips really happen at rate p
        info = {
            "intended_action": action,
            "actual_action": actual,
            "slipped": slipped,
            "on_ice": on_ice,
        }
        return self.pos, reward, terminated, truncated, info
    
    # ------------------------------------------------------------ display
    def render(self):
        """Return the map as text with the agent shown as A"""
        symbols = {FLOOR: ".", WALL: "#", ICE: "~", GOAL: "G"}
        lines=[]
        for r in range(self.grid.shape[0]):
            line=""
            for c in range(self.grid.shape[1]):
                line += "A" if (r,c)==self.pos else symbols[int(self.grid[r,c])]
            lines.append(line)
        return "\n".join(lines)