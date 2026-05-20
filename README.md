# Poker CFR – From Kuhn Poker to Texas Hold'em

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Counterfactual Regret Minimization (CFR & CFR+) implemented for Kuhn Poker.**  
A clean, educational codebase for learning imperfect‑information game solving.  
Future roadmap: scale to Leduc Hold'em → full Texas Hold'em using Deep CFR / NFSP.

## ✨ Features

- Full implementation of Kuhn Poker game rules (`game.py`)
- Tabular CFR (standard regret matching) – `cfr.py`
- Tabular CFR+ (positive regret accumulation) – `cfr_plus.py`
- Strategy logging & model saving (`utils.py`)
- Visualization tool for strategy convergence (`visualize.py`)
- Command‑line interface to choose algorithm and iterations (`trainer.py`)
- **Nash equilibrium** achieved in Kuhn Poker (game value ≈ −1/18)

## 📁 Project Structure

```text
├── game.py # Kuhn Poker rules, deck, payoff, terminal states
├── node.py # Regret and strategy storage (Node class)
├── cfr.py # Standard CFR algorithm
├── cfr_plus.py # CFR+ algorithm
├── trainer.py # Training loop with argparse support
├── utils.py # Save/load models and strategy snapshots
├── visualize.py # Plot strategy evolution from log files
├── models/ # Saved models (pickle files)
├── logs/ # Strategy snapshots (TXT)
└── visualizations/ # Generated strategy evolution plots
```


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib (for visualization)

Install dependencies:
```bash
pip install numpy matplotlib
```

### Training

Run training with default settings (CFR, 10 million iterations):

```bash
python trainer.py
```

Choose CFR+ and 1 million iterations:

```bash
python trainer.py --algo cfr_plus --iterations 1000000
```

Options:

`--algo / -a`: `cfr` or `cfr_plus`

`--iterations / -i`: integer (e.g. 10000000)

During training, every 100,000 iterations the average strategy is saved into `logs/strategy_{algo}_{total_iters}.txt`.

### Visualizing Strategy Convergence

After training, generate evolution plots:

```bash
python visualize.py
```

By default it reads the log file matching `strategy_1e+07.txt`.
You can modify the `plot_strategy_evolution` call inside visualize.py to point to other log files (e.g. `strategy_cfr_plus_1e+07.txt`).

Plots are saved under `visualizations/` with one PNG per information set.

## 📊 Algorithm Comparison

| Algorithm | Regret Update Rule               | Convergence Speed                        | Final Game Value (1e7 iters) |
|-----------|----------------------------------|------------------------------------------|------------------------------|
| CFR       | \( R_t = R_{t-1} + \Delta \)     | Standard                                 | −0.0560 (≈ −1/18)            |
| CFR+      | \( R_t = \max(0, R_{t-1} + \Delta) \) | Faster (fewer iterations to same quality) | −0.0556 (≈ −1/18)            |

Both converge to a Nash equilibrium of Kuhn Poker, but on different points of the equilibrium continuum.
CFR+ typically drives strategies closer to pure actions (e.g., King bets more often).

### Example Learned Strategy (CFR+ after 1M iterations)

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
Qpb: [0.4334, 0.5666]   # after check‑raise, call 57%
```

Game value: −0.0556 (theoretical Nash value is −1/18 ≈ −0.05556).

## 🗺️ Future Roadmap

- Leduc Hold’em (larger toy game) with card abstraction

- **Outcome Sampling CFR** (to handle huge game trees)

- **Deep CFR** / **Neural CFR **using PyTorch

- **NFSP** (Neural Fictitious Self‑Play) for full Hold'em

## 📖 References

- [Zinkevich et al. (2007) – "Regret Minimization in Games with Incomplete Information"](https://papers.nips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)

- Tammelin (2014) – "Solving Large Imperfect Information Games Using CFR+"

- Kuhn, H. W. (1950) – "A simplified two‑person poker"