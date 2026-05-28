# myCFR — Counterfactual Regret Minimization & RL for Poker

A collection of **tabular**, **neural**, and **RL-based** algorithms for solving imperfect-information poker games. Built on a shared game-abstraction layer supporting Kuhn Poker, Leduc Hold'em, Expanded Leduc, and River Poker.

---

## Quick Start

```bash
pip install numpy torch matplotlib

# Kuhn Poker — converges in seconds
python src/trainer.py -a cfr_plus -g kuhn -i 50000 --batch

# Leduc Hold'em — CFR+ batch+alternate (recommended)
python src/trainer.py -a cfr_plus -g leduc -i 2000000 --batch --alternate

# Expanded Leduc (4-rank) — variance-based checkpointing
python src/trainer.py -a cfr_plus -g expanded_leduc -i 2000000 --batch --alternate

# River Poker (2 hole cards, Hold'em-like) — range-vs-range bridge game
python src/trainer.py -a cfr_plus -g river_poker -i 1000000 --batch --alternate

# Neural / RL algorithms
python src/trainer.py -a nfsp_dual -g leduc -i 50000
python src/trainer.py -a deep_cfr_paper -g kuhn -i 2000
```

---

## Supported Games

| Game | Deck | Hole Cards | Rounds | Infosets | Tree Nodes | Nash (P0) |
|------|------|------------|--------|----------|------------|-----------|
| Kuhn Poker | 3 (J/Q/K) | 1 | 1 | 12 | ~20 | −1/18 ≈ −0.0556 |
| Leduc Hold'em | 6 (J/Q/K ×2) | 1 | 2 | 528 | 477 | ≈ −0.0855 |
| Expanded Leduc | 8 (J/Q/K/A ×2) | 1 | 2 | 928 | 631 | ≈ −0.099 (empirical) |
| River Poker | 8 (J/Q/K/A ×2) | **2** | 2 | ~1500 | 631 | unknown |

---

## CLI Reference

```bash
python src/trainer.py -a <algo> -g <game> -i <iterations> [--batch] [--alternate]
```

| Flag | Description |
|------|-------------|
| `-a, --algo` | `cfr`, `cfr_plus`, `dcfr`, `pdcfr_plus`, `mccfr`, `deep_cfr`, `deep_cfr_paper`, `dqn`, `ddqn`, `nfsp`, `nfsp_dual` |
| `-g, --game` | `kuhn`, `leduc`, `expanded_leduc`, `river_poker` |
| `-i, --iterations` | Iterations (tabular) or episodes (neural) |
| `--batch` | Enumerate all card-deal combos per iteration |
| `--alternate` | Alternate P0/P1 updates each iteration |

---

## Evaluation Pipeline

```bash
python eval_pipeline.py                      # scan models/, run all
python eval_pipeline.py --model leduc_cfr_plus_2e+06  # single model
python src/check_exploit.py leduc_cfr_plus_2e+06 --game leduc  # manual check
```

Output structure:
```
eval/{game}_{algo}_{iters}/
├── visualizations/
│   └── game_value.png
└── exploitability/
    └── {model_name}.txt
```

---

## Checkpoints

Two modes depending on whether Nash is known:

| Mode | Trigger | When |
|------|---------|------|
| Distance-to-Nash | `\|cur_value − nash\|` minimal | Nash known (Kuhn, Leduc) |
| Variance-based | Rolling-window std < 0.003 | Nash unknown (Expanded Leduc, River Poker) |

Output:
```
models/river_poker_cfr_plus_1e+06.pkl         ← final
models/river_poker_cfr_plus_best_3e+05.pkl    ← best checkpoint
```

---

## Algorithms

| # | Flag | Type | Key Idea |
|---|------|------|----------|
| 1 | `cfr` | Tabular | Regret matching |
| 2 | `cfr_plus` | Tabular | `max(0, regret + Δ)` + linear averaging |
| 3 | `dcfr` | Tabular | α/β/γ discounted regrets & strategy |
| 4 | `pdcfr_plus` | Tabular | Predictive R + discount + clamp |
| 5 | `deep_cfr` | Neural | RegretNet + reservoir buffer + alternating |
| 6 | `deep_cfr_paper` | Neural | External sampling + from-scratch retrain + LCFR |
| 7 | `dqn` | Neural | DQN (vs random) |
| 8 | `ddqn` | Neural | Double DQN (vs random) |
| 9 | `nfsp` | Neural | Single-sided NFSP |
| 10 | `nfsp_dual` | Neural | Bilateral NFSP (both players learn) |
| 11 | `mccfr` | Tabular | External Sampling MC-CFR |

---

## Project Structure

```
myCFR/
├── src/                        # Toy games (BFS-based)
│   ├── games/                  # Game ABC + kuhn, leduc, expanded_leduc, river_poker
│   ├── game_selector.py        # get_game(name)
│   ├── utils.py                # save/load model, log strategies
│   ├── algo/                   # All algorithms
│   │   ├── tabular/            # game_tree, node, cfr_tree
│   │   ├── mccfr/              # External Sampling MC-CFR
│   │   ├── neural/, deep_cfr/, dqn/, ddqn/, nfsp/, nfsp_dual/
│   ├── trainer.py              # Unified CLI + checkpoint logic
│   ├── check_exploit.py        # Exact exploitability (brute-force BR)
│   └── visualize.py
│
├── holdem/                     # Future: Texas Hold'em (on-the-fly + abstraction)
│   (planned: game.py, hand_eval.py, abstraction.py, tree.py, trainer.py)
│
├── eval_pipeline.py            # Auto-evaluation orchestrator
├── models/                     # Pickled strategies (*.pkl)
├── logs/                       # Strategy snapshots
├── eval/                       # Evaluation outputs
├── notprojectstuff/            # Reference docs (game rules, notes)
└── references/                 # PDF papers
```

---

## Hand Evaluation (River Poker)

Best 3-card hand from 2 hole + 1 community:

| Hand | Rank | Resolution |
|------|------|------------|
| Pair | 1 | Higher pair wins; kicker breaks ties |
| High card | 0 | Compare highest → second-highest |

---

## Training Commands

```bash
# ── Toy Games ──
python src/trainer.py -a cfr_plus -g kuhn -i 50000 --batch
python src/trainer.py -a cfr_plus -g leduc -i 2000000 --batch --alternate
python src/trainer.py -a cfr_plus -g expanded_leduc -i 2000000 --batch --alternate
python src/trainer.py -a cfr_plus -g river_poker -i 1000000 --batch --alternate

# ── Evaluation ──
python eval_pipeline.py
```
