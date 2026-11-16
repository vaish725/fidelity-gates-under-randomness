"""
Tests for data.py. Run from the project folder with:  python -m pytest -v
 
These use small datasets (n_starts=200) so the tests stay fast.
"""
 
import numpy as np
import pytest
 
from data import (behaviour_action, collect, distances_to_goal, load_dataset,
                  save_dataset, start_state)
from env import ICE, GridWorld
from oracle import rollout
 
 
def test_distances_are_shortest_paths():
    env = GridWorld()
    dist = distances_to_goal(env)
    assert dist[env.goal_pos] == 0
    assert dist[7, 6] == 1             # right next to G
    assert dist[0, 0] == -1            # a wall is never reached
    # Every free cell must be reachable, or the map has a sealed-off area.
    assert all(dist[cell] >= 0 for cell in env.free_cells())
 
 
def test_greedy_policy_always_reaches_goal_when_no_randomness():
    # With epsilon = 0 and p = 0, the policy should walk straight to G
    # from every cell, in exactly dist[cell] steps.
    env = GridWorld(p=0.0)
    dist = distances_to_goal(env)
    rng = np.random.default_rng(0)
    for cell in env.free_cells():
        if cell == env.goal_pos:
            continue
        env.reset(start=cell)
        steps = 0
        while not env.done:
            env.step(behaviour_action(env, dist, rng, epsilon=0.0))
            steps += 1
        assert env.pos == env.goal_pos and steps == dist[cell]
 
 
@pytest.fixture(scope="module")
def small():
    # A fixture builds something once and hands it to every test that asks for
    # it by name. scope="module" means: build once for this whole file.
    return {p: collect(p, seed=0, n_starts=200) for p in (0.0, 0.5)}
 
 
def test_shapes(small):
    d = small[0.5]
    H = int(d["horizon"])
    assert d["start_pos"].shape == (200, 2)
    assert d["actions"].shape == (200, H)
    assert d["real_paths"].shape == (200, H + 1, 2)
    # Each real path must begin at its start state.
    assert np.array_equal(d["real_paths"][:, 0], d["start_pos"])
 
 
def test_real_paths_only_move_one_cell_at_a_time(small):
    # A legal step changes row or column by at most 1 in total (or 0 at a wall).
    steps = np.abs(np.diff(small[0.5]["real_paths"], axis=1)).sum(axis=2)
    assert steps.max() <= 1
 
 
def test_p_zero_real_paths_match_replay(small):
    # Strongest consistency check: with no randomness, replaying the recorded
    # actions from the start state must reproduce the real path exactly.
    # This ties data.py, oracle.py and env.py together.
    d = small[0.0]
    env = GridWorld(p=0.0)
    for i in range(len(d["start_pos"])):
        replay = rollout(env, start_state(d, i), d["actions"][i], seed=0)
        assert np.array_equal(replay, d["real_paths"][i])
 
 
def test_ice_labels_are_correct(small):
    d = small[0.5]
    grid = GridWorld().grid
    for i in range(20):
        expected = sum(grid[r, c] == ICE for r, c in d["real_paths"][i, :-1])
        assert d["ice_steps"][i] == expected
 
 
def test_data_covers_ice_and_floor(small):
    # The practice-shift test needs a SPREAD of randomness in the pool: some
    # fully deterministic windows (0 ice steps) and some mostly-icy ones.
    # (Almost every 10-step window touches ice somewhere, because the only
    # route across the map goes through it, so "touches ice" alone is not a
    # useful yes/no label. The count of ice steps is.)
    ice = small[0.5]["ice_steps"]
    assert (ice == 0).mean() > 0.03       # some purely deterministic windows
    assert (ice >= 7).mean() > 0.10       # and a good share of very icy ones
    assert len(np.unique(ice)) >= 8       # with most levels in between present
 
 
def test_collection_is_reproducible():
    a = collect(0.3, seed=1, n_starts=100)
    b = collect(0.3, seed=1, n_starts=100)
    for key in a:
        assert np.array_equal(a[key], b[key])
 
 
def test_save_and_load_round_trip(small, tmp_path):
    # tmp_path is a pytest built-in: a fresh temporary folder, deleted afterwards.
    path = tmp_path / "test.npz"
    save_dataset(small[0.5], str(path))
    loaded = load_dataset(str(path))
    for key in small[0.5]:
        assert np.array_equal(loaded[key], small[0.5][key])
 