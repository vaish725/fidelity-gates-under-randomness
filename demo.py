"""
demo.py: see the gridworld in action.
Run with: python demo.py

Part 1 walks a planned route from S across the ice and prints the mao each step
Part 2 previews the core idea of H1: from the SAME start stae with the SAME
actions, how many different endings are there as p grows?
"""

import numpy as np

from env import ACTION_NAMES, DOWN, RIGHT, GridWorld
#a route from S at (1,1) to G at (7,7) that crosses the ice band:
#down 2 to row 3, right 5 along the ice to column 6, then down 4 to row 7, right 1.
ROUTE = [DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, DOWN, RIGHT]

# ---- Part 1: follow the route once with lots of ice randomness -----------
#seed=2 is picked because one slip sends the agent the wrong way, which is
#the interesting case to watch. Try other seeds to see other outcomes.
env = GridWorld(p=0.5, seed=2)
env.reset()
print("Start:\n"+ env.render()+"\n")
for a in ROUTE:
    pos, reward, terminated, truncated, info = env.step(a)
    note=""
    if info["slipped"] and info["actual_action"]!=a:
        note = f" -> SLIPPED, actually went {ACTION_NAMES[info['actual_action']]}"
    elif info["slipped"]:
        #a slip picks a random direction, and 1 time in 4 that's the one you chose.
        note = " -> slipped, but the random direction matched anyway"
    print(f"Chose {ACTION_NAMES[a]}{note}")
    print(env.render()+"\n")
    if env.done:
        print("Reached the goal!" if terminated else "Ran out of steps")
        break
else:
    #a for-loop's else runs only if we never hit 'break' i.e. never reached G
    print(f"Route finished at {env.pos} without reaching G (Slips pushed us off course).")

# ---- Part 2: same start, same actions, 50 different seeds -----------------
#start just above the ice and walk across it
start = (2,3)
actions = [DOWN, RIGHT, RIGHT, RIGHT, DOWN, DOWN]
print("\nEndings of the same 6 actions from", start, "replayed with 50 seeds:")
for p in [0.0,0.1,0.3,0.5]:
    env = GridWorld(p=p)
    endings=[]
    for seed in range(50):
        env.reset(start=start, seed=seed)
        for a in actions:
            env.step(a)
        endings.append(env.pos)
    #count how many different end cells there are, and how often the most
    #common one occurs. At p=0 there should be exactly one ending
    values, counts = np.unique(np.array(endings), axis=0, return_counts=True)
    print(f" p={p:.1f}: {len(values)} distinct endings, "
          f"most common ending reached {counts.max()}/50 times")