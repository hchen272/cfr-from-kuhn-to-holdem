# myCFR — Counterfactual Regret Minimization & RL for Poker

A collection of **tabular**, **neural**, and **RL-based** algorithms for solving imperfect-information poker games. Built on a shared game-abstraction layer supporting Kuhn Poker and Leduc Hold'em.

---

## Quick Start

```bash
pip install numpy torch matplotlib

# Kuhn Poker — converges in seconds
python src/trainer.py -a cfr_plus -g kuhn -i 50000 --batch

# Leduc Hold'em — CFR+ batch+alternate (recommended)
python src/trainer.py -a cfr_plus -g leduc -i 2000000 --batch --alternate

# Neural / RL algorithms
python src/trainer.py -a nfsp_dual -g leduc -i 50000
python src/trainer.py -a deep_cfr_paper -g kuhn -i 2000
```

---

## Evaluation Pipeline

```bash
python eval_pipeline.py                    # scan models/, run all
python eval_pipeline.py --model leduc_cfr_plus_2e+06  # single model
```

Auto-discovers `.pkl` models in `models/`, matches them with training logs, and produces:

```
eval/{game}_{algo}_{iters}/
├── visualizations/
│   └── game_value.png      # dual-line: cumulative avg + estimated current
└── exploitability/
    └── {model_name}.txt    # full BR report
```

`_best` checkpoint models are also detected and evaluated.

### Manual exploitability check

```bash
python src/check_exploit.py leduc_cfr_plus_2e+06 --game leduc
```

Works for all tabular CFR models and Deep CFR models (strategy saved in FakeNode format).

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

---

## CLI Reference

```bash
python src/trainer.py -a <algo> -g <game> -i <iterations> [--batch] [--alternate]
```

| Flag | Description |
|------|-------------|
| `-a, --algo` | Algorithm (see table above) |
| `-g, --game` | `kuhn` or `leduc` |
| `-i, --iterations` | Iterations (tabular) or episodes (neural) |
| `--batch` | Enumerate all (P0, P1, comm) combos per iteration |
| `--alternate` | Alternate P0/P1 updates each iteration |

---

## Checkpoints

All training loops automatically save the **best checkpoint** — the model whose rolling-window game value is closest to Nash. Output:

```
models/leduc_cfr_plus_2e+06.pkl           ← final
models/leduc_cfr_plus_best_5e+04.pkl      ← best checkpoint (iter in filename)
```

The checkpoint metric uses **rolling window average** (default: last 2% of iterations) rather than cumulative average, so early random strategies don't mask recent convergence.

---

## Project Structure

```
myCFR/
├── eval_pipeline.py          # Auto-evaluation orchestrator
├── diagnostic.py             # Cross-implementation comparison tool
│
├── src/
│   ├── games/               # Game ABC: kuhn.py, leduc.py
│   ├── game_selector.py     # get_game(name)
│   ├── utils.py             # save/load model, log strategies
│   │
│   ├── tabular/             # Game tree + CFR (cfr, cfr+, dcfr, pdcfr+)
│   ├── neural/              # Deep CFR (original)
│   ├── deep_cfr/            # Deep CFR (paper spec)
│   ├── dqn/                 # DQN
│   ├── ddqn/                # Double DQN
│   ├── nfsp/                # NFSP (single-sided)
│   ├── nfsp_dual/           # NFSP (bilateral) + dual-strategy logger
│   │
│   ├── trainer.py           # Unified CLI + checkpoint logic
│   ├── check_exploit.py     # Brute-force exploitability checker
│   ├── visualize.py         # Dual-line game value plots
│   └── diagnostic.py
│
├── rlcard_like/             # Reference Leduc + CFR (rlcard-compatible)
├── eval/                    # Evaluation outputs (model subdirs)
├── logs/                    # Strategy snapshots
├── models/                  # Pickled models
└── references/              # PDF papers
```

---

## Nash Values

| Game | Nash (P0) | Reference |
|------|-----------|-----------|
| Kuhn Poker | −1/18 ≈ −0.0556 | Kuhn 1950 |
| Leduc Hold'em | ≈ −0.0855 | Standard blind-based |

---

## Leduc Environment

- **Blinds**: P0 = SB (1 chip), P1 = BB (2 chips)
- **Rounds**: 2 rounds, fixed-limit (R1=2, R2=4)
- **Raise cap**: 1 bet + 2 raises per round (MAX_RAISES=3)
- **Deck**: 6 cards (J♠J♥ Q♠Q♥ K♠K♥), 528 infosets, 477 game-tree nodes
- **Payoffs**: normalised by BB (= 2), matching standard Leduc convention

Key fixes applied: blind-based investment tracking via `raised[]`, CALL always matches max bet, MAX_RAISES=3. These were the root cause of prior non-convergence.

---

## Training Commands

```bash
# ── Kuhn Poker ──
python src/trainer.py -a cfr_plus -g kuhn -i 50000 --batch

# ── Leduc Hold'em ──
python src/trainer.py -a cfr_plus -g leduc -i 2000000 --batch --alternate
python src/trainer.py -a dcfr -g leduc -i 5000000 --batch --alternate

# ── Evaluation ──
python eval_pipeline.py
```

---

## References

### Analysis

- [Kuhn Poker Equilibrium Analysis](comparison/kuhn_poker_equilibrium_analysis.md)
- [Leduc Hold'em Equilibrium Analysis](comparison/leduc_holdem_equilibrium_analysis.md)

### Papers
- Zinkevich et al. (2007) — Regret Minimization in Games with Incomplete Information
- Tammelin (2014) — Solving Large Imperfect Information Games Using CFR+
- Brown & Sandholm (2019) — Solving Imperfect-Information Games via Discounted Regret Minimization
- Brown et al. (2019) — Deep Counterfactual Regret Minimization
- Heinrich & Silver (2016) — Deep Reinforcement Learning from Self-Play (NFSP)
