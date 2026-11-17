# Fidelity gates under randomness

**Status: work in progress (week 1 of 3 complete).** No results yet.

## The question

World-model training methods like PROWL-2 use a *fidelity gate*: they compare one imagined rollout from the world model against the one real continuation that was recorded, and flag the start state if the two diverge. This treats any divergence as model error.

That assumption only holds when outcomes are deterministic. When the environment is genuinely random, a perfectly correct model can predict a different but equally valid outcome and still be flagged.

**H1:** as environment randomness increases, a single-reference gate raises more false alarms, and the states it keeps for training shift away from random situations.

This repo tests H1 in a small gridworld where the true distribution of outcomes is known exactly, so every gate decision can be scored as a false alarm or a real catch.

## Setup

- **Gridworld:** a 9×9 map with walls and a goal. A band of "ice" cells in the middle is the only route across. On ice, a move slips to a random direction with probability `p`; elsewhere the world is deterministic. `p` is swept over {0, 0.1, 0.3, 0.5}.
- **Oracle:** restores any start state and replays the recorded actions under 50 seeds, giving the true spread of outcomes.
- **Data:** an ε-greedy behaviour policy (ε = 0.3) produces 1,000 recorded windows per `p` and per seed (3 seeds). Each window holds a start state, 10 recorded actions, and the one real path that followed.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | All experiment settings and the seed plan |
| `env.py` | The gridworld (the "real environment") |
| `oracle.py` | Multi-seed replays giving the true outcome distribution |
| `data.py` | Behaviour policy and recorded windows |
| `demo.py` | Visual demo of slipping on the ice |
| `tests/` | Tests for every module |

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v          # run the tests
python demo.py     # see the gridworld in action
python data.py     # generate the datasets (written to datasets/)
```

## Roadmap

- [x] Gridworld with tunable randomness, plus tests
- [x] Oracle (exact replay of the true outcome distribution)
- [x] Behaviour policy and recorded data
- [x] World models: one correct, plus planted errors (structural and calibration)
- [ ] Single-reference gate (PROWL-2-style) and an oracle gate as the upper bound
- [ ] Sweep over `p`, thresholds and seeds: false-alarm rate, catch rate, AUROC, practice shift
- [ ] Plots, write-up and limitations

## References

- Odyssey. *PROWL-2: Jointly learning simulation and decision-making*, 2026.
- Güzel et al. *PROWL-1*, arXiv 2605.18803, 2026.
- Jiang et al. *Grounding aleatoric uncertainty for unsupervised environment design*, NeurIPS 2022.
- Sadeghi et al. *CaliBench*, TMLR 2026.