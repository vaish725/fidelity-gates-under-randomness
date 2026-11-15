"""
Tests for oracle.py.
Run from the project folder with: python -m pytest -v
"""
import copy
import numpy as np
import pytest
from env import DOWN, LEFT, RIGHT, UP, EnvState, GridWorld
from oracle import end_distribution, oracle_rollouts, rollout

ICE_START = EnvState(pos=(4,4), t=0, done=False)

def test_p_zero_has_one_outcome():
    #with no randomness, all 50 replays must agree: one ending, probability 1
    #path by hand: (4,4) -UP-> (3,4) -UP-> wall at (2,4), stay (3,4)
    #               -RIGHT-> (3,4) -RIGHT-> (3,6)
    env = GridWorld(p=0.0)
    paths = oracle_rollouts(env, ICE_START, [UP, UP, RIGHT, RIGHT])
    assert end_distribution(paths)=={(3,6): 1.0}


@pytest.mark.parametrize("p", [0.1,0.3,0.5])
def test_one_ice_step_matches_hand_calculations(p):
    #the oracle's odds for one step must match the maths in env.py:
    #intended cell 1-0.75p, each other neighbour p/4
    env = GridWorld(p=p)
    dist = end_distribution(oracle_rollouts(env, ICE_START, [UP], n_seeds=8000))
    expected = {(3,4): 1 - 0.75 * p, (4,5): p / 4, (5,4): p / 4, (4,3): p / 4}
    for pos, prob in expected.items():
        #0.02 is several standard errors at 8,000 replays
        assert abs(dist.get(pos, 0.0) - prob)<0.02

def test_shapes_and_padding_at_goal():
    # From next to G, the first action reaches the goal; the remaining 4
    # steps should be padded with the goal position.
    env = GridWorld(p=0.0)
    start = EnvState(pos=(7, 6), t=0, done=False)
    actions = [RIGHT, DOWN, DOWN, DOWN, DOWN]
    paths = oracle_rollouts(env, start, actions, n_seeds=3)
    assert paths.shape == (3, len(actions) + 1, 2)   # (seeds, steps + 1, row/col)
    assert (paths[:, 1:] == [7, 7]).all()             # stuck at the goal after step 1


def test_oracle_is_reproducible():
    # Same base_seed -> the exact same 50 paths. Needed for repeatable results.
    a = oracle_rollouts(GridWorld(p=0.5), ICE_START, [UP, LEFT, DOWN], base_seed=7)
    b = oracle_rollouts(GridWorld(p=0.5), ICE_START, [UP, LEFT, DOWN], base_seed=7)
    assert np.array_equal(a, b)


def test_oracle_leaves_environment_untouched():
    # Calling the oracle must not change the caller's position OR their
    # random number stream.
    env = GridWorld(p=0.5, seed=1)
    env.reset(start=(2, 2))
    state_before = env.get_state()
    rng_twin = copy.deepcopy(env.rng)       # an identical copy to compare against

    oracle_rollouts(env, ICE_START, [UP, RIGHT, DOWN])

    assert env.get_state() == state_before
    assert env.rng.random() == rng_twin.random()  # the next random number is unchanged


def test_rollout_matches_live_play():
    # rollout() should give exactly the path you'd get by stepping by hand
    # with the same seed, so the oracle and the "real" data are consistent.
    actions = [DOWN, RIGHT, RIGHT, RIGHT, DOWN, DOWN]
    start = EnvState(pos=(2, 3), t=0, done=False)

    live = GridWorld(p=0.5)
    live.reset(start=start.pos, seed=99)
    live_path = [start.pos] + [live.step(a)[0] for a in actions]

    replay = rollout(GridWorld(p=0.5), start, actions, seed=99)
    assert [tuple(x) for x in replay.tolist()] == live_path
