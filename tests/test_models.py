"""
Tests for models.py
Run from the project folder with: pytest -v
"""

import numpy as np
import pytest

from data import collect, start_state
from env import DOWN, RIGHT, UP, WALL, EnvState, GridWorld
from models import MODEL_NAMES, make_model, tv_distance
from oracle import end_distribution, oracle_rollouts, rollout

def test_correct_model_matches_real_world_exactly():
    #same rules + same seed -> identical paths, on real recorded windows.
    data = collect(0.5, seed=0, n_starts=100)
    real, model = GridWorld(p=0.5), make_model("correct", 0.5)
    for i in range(100):
        s, a = start_state(data, i), data["actions"][i]
        assert np.array_equal(rollout(real, s, a, seed=i), rollout(model, s, a, seed=i))

def test_missing_wall_walks_through_wall():
    start = EnvState(pos=(5,4), t=0, done=False)
    assert rollout(GridWorld(p=0.0), start, [DOWN], seed=0)[-1].tolist() == [5,4]
    assert rollout(make_model("missing_wall", 0.0), start, [DOWN], seed=0)[-1].tolist() == [6,4]


def test_missing_wall_does_not_change_real_world():
    #building a buggy model must never eidt the real environment's map
    make_model("missing_wall", 0.0)
    assert GridWorld().grid[6,4]==WALL

def test_teleport_jumps():
    start = EnvState(pos=(5,1), t=0, done=False) #just left of (5,2)
    end = rollout(make_model("teleport", 0.0), start, [RIGHT], seed=0)[-1]
    assert end.tolist() == [1,6]


def test_overconfident_never_slips():
    model = make_model("overconfident", 0.5)
    for seed in range(500):
        model.reset(start=(4,4), seed=seed)
        assert not model.step(UP)[4]["slipped"]


def test_bugs_change_nothing_away_from_the_bug():
    # Common random numbers: on a window that can't reach either bug cell
    # (too far away in 3 steps), buggy models must match the real world
    # exactly, seed for seed, even with heavy randomness.
    start = EnvState(pos=(3,5), t=0, done=False)
    actions = [RIGHT, UP, UP]
    for name in ("missing_wall","teleport"):
        for seed in range(200):
            real = rollout(GridWorld(p=0.5), start, actions, seed)
            pred = rollout(make_model(name, 0.5), start, actions, seed)
            assert np.array_equal(real, pred)


def test_tv_distance_basics():
    a = {(1,1): 0.5, (2,2): 0.5}
    assert tv_distance(a,a) == 0.0 #identical
    assert tv_distance(a, {(3,3): 1.0}) == 1.0 #no overlap
    assert tv_distance(a, {(1,1): 1.0}) == pytest.approx(0.5) #half overlap


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_is_correct_when_bug_is_out_of_reach(name):
    # at p = 0 on a short floor-only window, every model (even the buggy ones)
    # gives exactly the true outcome. A bug is only an error where it matters.
    start = EnvState(pos=(1,1), t=0, done=False)
    truth = end_distribution(oracle_rollouts(GridWorld(p=0.0), start, [DOWN, RIGHT]))
    pred = end_distribution(oracle_rollouts(make_model(name, 0.0), start, [DOWN, RIGHT]))
    assert tv_distance(pred, truth) == 0.0
    