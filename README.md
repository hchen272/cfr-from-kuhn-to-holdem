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
python src/trainer.py -a ddqn      -g kuhn -i 100000
python src/trainer.py -a nfsp_dual -g leduc -i 50000
python src/trainer.py -a deep_cfr_paper -g kuhn -i 2000
```

---

## Evaluate Exploitability (tabular models only)

```bash
python src/check_exploit.py leduc_cfr_plus_2e+06 --game leduc
python src/check_exploit.py kuhn_cfr_1e+07 --game kuhn
```

Outputs: BR values, per-player regrets (ε₀, ε₁), exploitability = (ε₀ + ε₁) / 2.

**Note**: Game tree structure changed in the blind-based Leduc fix — old `.pkl` models are incompatible. Delete `models/leduc_*` before training.

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
| 7 | `dqn` | Neural | DQN (vs random opponent) |
| 8 | `ddqn` | Neural | Double DQN (vs random opponent) |
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

## Project Structure

```
myCFR/
├── src/
│   ├── games/               # Game ABC: kuhn.py, leduc.py
│   ├── game_selector.py     # get_game(name)
│   ├── utils.py             # save/load model, log strategies
│   │
│   ├── tabular/             # Game tree + CFR (cfr, cfr+, dcfr, pdcfr+)
│   ├── neural/              # Deep CFR (original)
│   ├── deep_cfr/            # Deep CFR (paper spec, Brown 2019)
│   ├── dqn/                 # DQN
│   ├── ddqn/                # Double DQN
│   ├── nfsp/                # NFSP (single-sided)
│   ├── nfsp_dual/           # NFSP (bilateral) + dual-strategy logger
│   │
│   ├── trainer.py           # Unified CLI entry point
│   ├── check_exploit.py     # Brute-force exploitability checker
│   ├── visualize.py         # Auto-scan logs → plots
│   └── diagnostic.py        # Cross-implementation comparison tool
│
├── rlcard_like/             # Reference Leduc + CFR (rlcard-compatible)
│   ├── games/leducholdem/
│   ├── agents/cfr_agent.py
│   └── train.py
│
├── logs/                    # Strategy snapshots
├── models/                  # Pickled models
├── visualizations/          # Per-algo plots
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

Key fixes applied: blind-based investment tracking via `raised[]`, CALL always matches current max bet, MAX_RAISES=3 for standard rules. These were the root cause of prior non-convergence.

---

## Recommended Training Commands

```bash
# ── Kuhn Poker ──
python src/trainer.py -a cfr_plus -g kuhn -i 50000 --batch

# ── Leduc Hold'em ──
python src/trainer.py -a cfr_plus -g leduc -i 2000000 --batch --alternate
python src/trainer.py -a dcfr -g leduc -i 5000000 --batch --alternate
python src/trainer.py -a cfr -g leduc -i 1000000 --batch

# ── Neural exploration ──
python src/trainer.py -a nfsp_dual -g leduc -i 50000
python src/trainer.py -a deep_cfr -g kuhn -i 1000000 --alternate

# ── External reference ──
python rlcard_like/train.py -i 100000

# ── Cross-implementation diagnostic ──
python diagnostic.py
```

---

## References

- Zinkevich et al. (2007) — Regret Minimization in Games with Incomplete Information
- Tammelin (2014) — Solving Large Imperfect Information Games Using CFR+
- Brown & Sandholm (2019) — Solving Imperfect-Information Games via Discounted Regret Minimization
- Brown et al. (2019) — Deep Counterfactual Regret Minimization
- Heinrich & Silver (2016) — Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP)
