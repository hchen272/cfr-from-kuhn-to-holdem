# Poker CFR – From Kuhn Poker to Texas Hold'em

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Counterfactual Regret Minimization (CFR) family for Kuhn Poker and Leduc Hold'em.**  
A clean, educational codebase for learning imperfect‑information game solving.
Includes tabular algorithms (CFR, CFR+, DCFR, PDCFR+) and a neural variant
(Deep CFR), all built on a **game-agnostic** abstraction layer supporting
multiple poker games.

## Features

- **2 games**: Kuhn Poker (`games/kuhn.py`) + Leduc Hold'em (`games/leduc.py`)
- Game abstraction layer (`games/`, `game_selector.py`) — easy to add new games
- **Pre-computed game tree** (`tabular/game_tree.py`) — integer-encoded tree with payoff cache, **~3× speedup** over string-based traversal
- **5 tabular algorithms** (two implementations each):
  - `cfr` — Standard CFR (regret matching)
  - `cfr_plus` — CFR+ (positive regret accumulation)
  - `dcfr` — Discounted CFR (separate α/β/γ discounts)
  - `pdcfr_plus` — Predictive Discounted CFR+ (prediction + discount + clamp)
- **Batch mode** (`--batch`) — enumerates all (P0,P1,comm) combos per iteration for 18× more data per outer loop
- **Alternating updates** (`--alternate`) — P0/P1 swap each iteration
- 1 **neural variant**: Deep CFR (`neural/`, PyTorch)
- Strategy logging & model saving (`utils.py`)
- **Exploitability checker** (`check_exploit.py`) — brute-force Nash distance
- Self-contained visualisation tool (`visualize.py`)
- Checkpoint-based best-strategy recovery (Deep CFR)
- **Kuhn Poker**: Nash equilibrium achieved by all algorithms (game value ≈ −1/18)
- **Leduc Hold'em**: Implemented; known slow convergence (needs 10⁸+ iterations)

## Project Structure

```text
myCFR/
├── logs/                        # Training logs (e.g. kuhn_strategy_cfr_1e+07.txt)
├── models/                      # Saved model pickles (e.g. kuhn_cfr_1e+07.pkl)
├── visualizations/              # Per-algorithm plots (e.g. kuhn_cfr_1e+07/)
├── comparison/                  # Detailed equilibrium analysis
│   └── kuhn_poker_equilibrium_analysis.md
│
├── src/
│   ├── games/                   # Game abstraction layer
│   │   ├── __init__.py          # Game ABC (abstract base class)
│   │   ├── kuhn.py              # KuhnGame(Game) — 12 infosets
│   │   └── leduc.py             # LeducGame(Game) — ~620 infosets
│   ├── game_selector.py         # get_game(name) → Game instance
│   ├── game.py                  # Backward-compat re-export
│   │
│   ├── tabular/                 # Pre-computed game tree + tree-based CFR
│   │   ├── game_tree.py         # GameTree BFS builder, payoff cache, IID map
│   │   ├── node.py              # Node class (regret/strategy storage)
│   │   └── cfr_tree.py          # 4 tree-based CFR functions
│   │
│   ├── neural/                  # Neural CFR series
│   │   ├── model.py             # RegretNet (PyTorch)
│   │   ├── buffer.py            # Reservoir buffer
│   │   ├── deep_cfr.py          # Deep CFR traversal
│   │   └── train.py             # Training loop
│   │
│   ├── trainer.py               # Unified CLI entry point
│   ├── check_exploit.py         # Brute-force exploitability checker
│   └── visualize.py             # Auto-scan logs → plots
│
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib (for visualisation)
- PyTorch (for Deep CFR only)

Install dependencies:

```bash
pip install numpy matplotlib
# for Deep CFR:
pip install torch
```

### Training

Run from the project root. Use `--game` / `-g` to select the game
(default `kuhn`):

```bash
# ── Kuhn Poker (converges at 10⁷ iters, Nash ≈ −1/18) ──
python src/trainer.py                         # CFR, Kuhn, 10M iters
python src/trainer.py -a cfr_plus             # CFR+, Kuhn, 10M iters
python src/trainer.py -a dcfr                 # DCFR, Kuhn, 10M iters
python src/trainer.py -a pdcfr_plus           # PDCFR+, Kuhn, 10M iters
python src/trainer.py -a deep_cfr             # Deep CFR, Kuhn, 1M iters

# ── Leduc Hold'em (needs 10⁸+ iters, Nash ≈ −0.085) ──
python src/trainer.py -a cfr -g leduc -i 10000000
python src/trainer.py -a cfr_plus -g leduc -i 10000000 --batch
python src/trainer.py -a pdcfr_plus -g leduc -i 1000000 --batch --alternate
python src/trainer.py -a deep_cfr -g leduc -i 1000000

# ── Evaluate strategy quality (exploitability) ──
python src/check_exploit.py leduc_cfr_plus_5e+06 --game leduc
```

Options:

| Flag | Description |
|------|-------------|
| `--algo` / `-a` | Algorithm: `cfr`, `cfr_plus`, `dcfr`, `pdcfr_plus`, `deep_cfr` |
| `--iterations` / `-i` | Number of iterations (default 10⁷ tabular, 10⁶ deep) |
| `--game` / `-g` | Game: `kuhn` (default) or `leduc` |
| `--batch` | Enumerate all (P0,P1,comm) combos per iteration (18× for Leduc) |
| `--alternate` | Alternating P0/P1 updates each iteration |

During training, snapshots are saved every 1% of total iterations into `logs/`
as `{game_name}_strategy_{algo}_{iters}.txt`. Models are saved to `models/`
as `{game_name}_{algo}_{iters}.pkl`.

### Visualising Convergence

```bash
python src/visualize.py
```

Automatically discovers all `*_strategy_*.txt` log files (for any game),
generates plots under `visualizations/{game_name}_{algo}_{iters}/`:
- **Kuhn**: 12 per-infoset strategy PNGs + 1 game value curve
- **Leduc**: only the game value curve (too many infosets for per-infoset plots)

## Algorithm Comparison (Kuhn Poker)

| Algorithm | Type | Regret Update | Deep CFR Checkpoint | Final Game Value |
|-----------|------|---------------|---------------------|------------------|
| CFR | Tabular | $R_t = R_{t-1} + \Delta$ | — | −0.0560 (≈ −1/18) |
| CFR+ | Tabular | $R_t = \max(0, R_{t-1} + \Delta)$ | — | −0.0556 (≈ −1/18) |
| DCFR | Tabular | $R_t = \alpha [R_{t-1}]_+ + \beta [R_{t-1}]_- + \Delta$ | — | −0.0552 (≈ −1/18) |
| PDCFR+ | Tabular | Predictive $R_{t-1} + \Delta$ + discount + clamp | — | −0.0553 (≈ −1/18) |
| **Deep CFR** | **Neural** | **Function approx. + replay buffer** | **−0.0527** | **−0.0556** (target) |

All tabular algorithms converge to the Nash equilibrium of Kuhn Poker. Deep
CFR approximates the equilibrium; its best checkpoint reaches above the
theoretical Nash value (−0.0527 > −0.0556).

### Example Learned Strategies (10⁷ iterations)

#### CFR+ (tabular)

```text
J:   [0.7678, 0.2322]   # Jack: 23% bluff when first to act
Jp:  [0.6671, 0.3329]   # facing a check, bluff 1/3
Jb:  [1.0,   0.0]       # facing a bet, always fold
K:   [0.3018, 0.6982]   # King: bet 70% of the time
Kp:  [0.0,   1.0]       # facing a check, always bet
Kb:  [0.0,   1.0]       # facing a bet, always call
Q:   [0.9924, 0.0076]   # Queen: almost always check
Qp:  [1.0,   0.0]       # facing a check, check
Qb:  [0.6651, 0.3349]   # facing a bet, call 1/3
Qpb: [0.4334, 0.5666]   # after check-raise, call 57%
```

Game value: −0.0556 (theoretical Nash value is −1/18 ≈ −0.05556).

#### Deep CFR (best checkpoint, 830k iters)

```text
J:   [0.7732, 0.2268]   # Jack: 23% bluff — matches tabular
Jb:  [0.9764, 0.0236]   # fold 98% — near perfect
Jp:  [0.5868, 0.4132]   # bluff 41% when checked to
K:   [0.2976, 0.7024]   # King: bet 70% — matches tabular
Kb:  [0.0228, 0.9772]   # call 98% — near perfect
Kp:  [0.0255, 0.9745]   # bet 97% — near perfect
Q:   [0.8174, 0.1826]   # Queen: 18% bet (still too high)
Qb:  [0.5177, 0.4823]   # call 48% (should be 33%)
Qp:  [0.8516, 0.1484]   # 15% bet after check (too high)
Qpb: [0.4240, 0.5760]   # call 58% — matches tabular
```

Checkpoint game value: −0.0527. The network learns J and K well; Queen
aggression is the main remaining gap.

## Detailed Analysis

See [Kuhn Poker Equilibrium Analysis](comparison/kuhn_poker_equilibrium_analysis.md)
for a thorough per-infoset breakdown, convergence dynamics, and training
oscillation analysis.

## Roadmap

- [x] Game abstraction layer (`games/`, `game_selector.py`)
- [x] Game-agnostic CFR algorithms (all 5 variants accept `game` parameter)
- [x] Dynamic file naming (`{game_name}_strategy_*`, `{game_name}_model_*`)
- [x] Leduc Hold'em game implementation (6 cards, 2 rounds, community card)
- [ ] Large-scale Leduc training & convergence verification (10⁸+ iters)
- [ ] Leduc equilibrium analysis (similar to Kuhn comparison doc)
- [ ] Outcome Sampling CFR (to handle huge game trees)
- [ ] Neural Fictitious Self-Play (NFSP) for full Hold'em

## References

- Zinkevich et al. (2007) – "Regret Minimization in Games with Incomplete Information"
- Tammelin (2014) – "Solving Large Imperfect Information Games Using CFR+"
- Brown & Sandholm (2019) – "Solving Imperfect-Information Games via Discounted Regret Minimization"
- Brown et al. (2019) – "Deep Counterfactual Regret Minimization"
- Kuhn, H. W. (1950) – "A simplified two‑person poker"
