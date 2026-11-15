"""
Tests for env.py 
Run from project folder with: python -m pytest -v

Each test checks one promise the experiment relies on. If any of these were wrong, 
the H1 results would be wrong too, so its worth checking them first.
"""

import numpy as np
import pytest

from env import DOWN, RIGHT, LEFT, UP, GridWorld

# a cell in the middle of the ice, and a plain floor cell, from DEFAULT_MAP
ICE_CELL = (4,4)
FLOOR_CELL=(2,2)

def run(env, actions):
    """Apply a list of actions and return the list of positions visited."""
    path=[]
    for a in actions:
        pos, _, _, _, _ = env.step(a)
        path.append(pos)
        if env.done:
            break
        return path
    
def test_p_zero_is_deterministic():
    #with p=0 the seed must not matter: the same actions always give the same path
    actions = [DOWN, DOWN, RIGHT, RIGHT, DOWN, RIGHT, RIGHT, DOWN]
    paths=[]
    for seed in range(20):
        env = GridWorld(p=0.0, seed=seed)
        env.reset()
        paths.append(run(env, actions))
    assert all(path==paths[0] for path in paths)


def test_walls_block_movement():
    #from S at (1,1), moving UP or LEFT hits the outer wall, so we stay put
    env = GridWorld(p=0.0, seed=0)
    env.reset()
    assert env.step(UP)[0]==(1,1)
    assert env.step(LEFT)[0]==(1,1)


def test_no_slips_off_ice():
    #even with heavy randomness, floor cells must stay deterministic
    env = GridWorld(p=0.9, seed=0)
    for _ in range(2000):
        env.reset(start = FLOOR_CELL)
        _, _, _, _, info = env.step(RIGHT)
        assert not info["slipped"]
        assert info["actual_action"]==RIGHT


@pytest.mark.parametrize("p", [0.1,0.3,0.5])
def test_slip_rate_on_ice_matches_p(p):
    #on ice, a slip should happen with probability p, and the move should go
    #somewhere unintended with probability p*3/4 (a slip picks the intended 
    #direction again 1 time in 4)
    env = GridWorld(p=p, seed=123)
    n=20_000
    slips = wrong_dir = 0
    for _ in range(n):
        env.reset(start=ICE_CELL)
        _, _, _, _, info = env.step(UP)
        slips += info["slipped"]
        wrong_dir += info["actual_action"] != UP
    #0.015 is about 4-5 std errors at n=20,000: lose enough to never
    #fail by bad luck, tight enough to catch a real bug
    assert abs(slips/n-p)<0.015
    assert abs(wrong_dir/n-0.75*p)<0.015


def test_save_and_restore_replays_exactly():
    #same saved state + same seed + same actions must give an identical path
    #this is what lets the oracle trust its replays
    env = GridWorld(p=0.5, seed=0)
    env.reset(start=ICE_CELL)
    saved = env.get_state()
    actions = [UP, LEFT, DOWN, DOWN, RIGHT, UP, RIGHT, DOWN]

    env.rng=np.random.default_rng(42)
    first = run(env, actions)

    env.set_state(saved)
    env.rng=np.random.default_rng(42)
    second = run(env, actions)

    assert first==second


def test_different_seeds_differ_on_ice():
    #the flip side: with p>0, replays from ice under different seeds should
    #NOT all be identical. Otherwise there is no randomness to study.
    env = GridWorld(p=0.5)
    saved_paths = set()
    for seed in range(50):
        env.reset(start=ICE_CELL, seed=seed)
        saved_paths.add(tuple(run(env,[UP, UP, LEFT, DOWN])))
    assert len(saved_paths)>1


def test_goal_ends_episode_with_reward():
    env = GridWorld(p=0.0, seed=0)
    env.reset(start=(7,6)) #the cell just left of G at (7,7)
    pos, reward, terminated, truncated, _ = env.step(RIGHT)
    assert pos == (7,7) and reward == 1.0 and terminated and not truncated
    with pytest.raises(RuntimeError): #stepping after the end is a mistake
        env.step(RIGHT)


def test_truncation_at_max_steps():
    env = GridWorld(p=0.0, max_steps=3, seed=0)
    env.reset()
    for _ in range(2):
        assert not env.step(UP)[3] #truncated is index=3; not yet
    assert env.step(UP)[3] #third step hits the limit