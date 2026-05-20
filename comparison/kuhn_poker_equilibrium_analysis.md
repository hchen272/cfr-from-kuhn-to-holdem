# Kuhn Poker Equilibrium Analysis

A comparative analysis of **five** CFR variants (CFR, CFR+, DCFR, PDCFR+, Deep CFR)
trained on Kuhn Poker. Tabular algorithms use 10⁷ iterations; Deep CFR uses 10⁶
iterations with checkpoint-based best-strategy recovery.

---

## Notation

Kuhn Poker is a two-player, zero-sum game with a 3-card deck (J < Q < K).
Each player antes 1 chip, receives one private card, and acts in a
pass/bet sequence with at most one bet per player.

Each **infoset** is labelled as `{card}{betting_history}`:

| Infoset | Meaning | Action 0 | Action 1 |
|---------|---------|----------|----------|
| `J` / `Q` / `K` | First to act (player 0) | Check | Bet |
| `Jp` / `Qp` / `Kp` | Second to act (player 1), facing a check | Check | Bet |
| `Jb` / `Qb` / `Kb` | Second to act (player 1), facing a bet | Fold | Call |
| `Jpb` / `Qpb` / `Kpb` | First to act back (player 0), facing a bet after a check | Fold | Call |

Strategy arrays are **[P(Action 0), P(Action 1)]** — the probability of each action.

The **game value** is from player 0's perspective. A negative value indicates
player 0 is at a disadvantage (expected to lose ~1/18 chip per hand), which
matches theory since acting first leaks information.

---

## 1. Overall Convergence

| Algorithm | Game Value | vs. Nash (−1/18 ≈ −0.05556) | Iterations |
|-----------|------------|------------------------------|------------|
| CFR       | −0.0560    | −0.00044 (within noise)      | 10⁷        |
| CFR+      | −0.0556    | ≈ 0 (exact match)            | 10⁷        |
| DCFR      | −0.0552    | +0.00036 (within noise)      | 10⁷        |
| PDCFR+    | −0.0553    | +0.00026 (within noise)      | 10⁷        |
| **Deep CFR** | **−0.0527** | **+0.00286** (checkpoint)  | **10⁶**    |

All four tabular algorithms converge to the Nash equilibrium within sampling
noise. Deep CFR achieves a checkpoint average value of −0.0527 (above the
theoretical Nash), suggesting the network's best-policy performance is near
equilibrium despite oscillation in the training dynamics.

---

## 2. Full Strategy Comparison

Deep CFR data are from the **best checkpoint** (restored at 830 000 iterations,
average game value −0.0527).

### 2.1 Jack (J) — the bluffing hand

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | **Deep CFR** | Interpretation |
|---------|-----|------|------|--------|--------------|----------------|
| **J**   | [0.816, 0.184] | [0.768, 0.232] | [0.742, 0.258] | [0.772, 0.228] | **[0.773, 0.227]** | Bluff-bet 23% — matches CFR+ and PDCFR+ |
| **Jp**  | [0.667, 0.333] | [0.667, 0.333] | [0.665, 0.335] | [0.667, 0.333] | **[0.587, 0.413]** | Bluff more when checked to (41%) |
| **Jb**  | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | **[0.976, 0.024]** | Near-perfect fold (2% call noise) |
| **Jpb** | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | **[0.985, 0.015]** | Near-perfect fold (1.5% call noise) |

Deep CFR learns the correct bluff frequency for Jack when first to act (23%).
The bluff frequency when checked to (Jp: 41%) is higher than the tabular ≈33%,
indicating the neural network has not fully converged at the checkpoint.

### 2.2 King (K) — the value hand

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | **Deep CFR** | Interpretation |
|---------|-----|------|------|--------|--------------|----------------|
| **K**   | [0.450, 0.550] | [0.302, 0.698] | [0.229, 0.771] | [0.317, 0.683] | **[0.298, 0.702]** | Bet 70% — matches CFR+ closely |
| **Kp**  | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **[0.026, 0.974]** | Near-perfect (2.5% check noise) |
| **Kb**  | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **[0.023, 0.977]** | Near-perfect (2.3% fold noise) |
| **Kpb** | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **[0.038, 0.962]** | Near-perfect (3.8% fold noise) |

King strategies are very close to the tabular algorithms. The small residual
noise (2–4%) is characteristic of neural approximation — the network rarely
hits exactly 0% or 100%.

### 2.3 Queen (Q) — the bluff-catcher

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | **Deep CFR** | Interpretation |
|---------|-----|------|------|--------|--------------|----------------|
| **Q**   | [1.0, ≈0] | [0.999, 0.001] | [1.0, ≈0] | [0.999, 0.001] | **[0.817, 0.183]** | **18% bet** — still too aggressive |
| **Qp**  | [1.0, ≈0] | [1.0, ≈0] | [1.0, ≈0] | [1.0, ≈0] | **[0.852, 0.148]** | **15% bet** after check — too aggressive |
| **Qb**  | [0.666, 0.334] | [0.665, 0.335] | [0.667, 0.333] | [0.665, 0.335] | **[0.518, 0.482]** | Call 48% — too high (should be 33%) |
| **Qpb** | [0.482, 0.518] | [0.433, 0.567] | [0.410, 0.590] | [0.439, 0.561] | **[0.424, 0.576]** | Call 58% — matches the tabular range |

The Queen is the hardest for Deep CFR. The network still bets Queen ~18% of
the time when it should check almost always. This is the primary remaining gap
between Deep CFR and the tabular algorithms.

However, **Qpb** (call after check-raise) at 58% is well within the tabular
range (52–59%), showing that the network understands the bluff-catching
dynamics in this subgame.

---

## 3. Algorithm Behavioural Differences

### 3.1 Strategy Comparison

| Algorithm | Type | Learning | Strengths | Weaknesses |
|-----------|------|----------|-----------|------------|
| CFR | Tabular | Exact regret matching | Most theoretically sound, predictable convergence | Slowest convergence |
| CFR+ | Tabular | Positive regret clamp | Fast tabular convergence, exact Nash at 10⁷ | Slightly more aggressive than CFR |
| DCFR | Tabular | Discounted regrets (α, β, γ) | Fastest tabular convergence, most decisive strategies | Most extreme (furthest from uniform) |
| PDCFR+ | Tabular | Predictive + discounted + clamped | Best early convergence among tabular | Adds complexity for marginal gain on Kuhn |
| **Deep CFR** | **Neural** | **Function approximation + replay** | **Generalises to large games; no explicit node table** | **Higher variance; residual noise in pure actions** |

### 3.2 Equilibrium Continuum

Kuhn Poker has a **continuum of Nash equilibria**, not a single unique one.
The different algorithms converge to different points:

| Decision point | CFR | CFR+ | DCFR | PDCFR+ | **Deep CFR** |
|----------------|-----|------|------|--------|--------------|
| K-bet (first act) | 55% | 70% | **77%** | 68% | **70%** |
| J-bluff (first act) | 18% | 23% | **26%** | 23% | **23%** |
| Q-bet (first act) | ≈0% | ≈0% | ≈0% | ≈0% | **18%** ❌ |
| Qb-call (facing bet) | 33% | 33% | 33% | 34% | **48%** |
| Qpb-call (check-call) | 52% | 57% | **59%** | 56% | **58%** |

### 3.3 Deep CFR Training Dynamics

Deep CFR exhibits a distinctive **oscillation pattern** not seen in tabular
algorithms:

```
10k → −0.3064  (random start)
50k → −0.1733  (fast initial improvement)
...
160k → −0.0569 ★ first approach to Nash
170k → −0.0549 ★ above Nash
...
800k → −0.0537 ★ second peak (better than first)
830k → −0.0527 ★ best checkpoint
...
1M  → −0.0691  (oscillated back)
```

The network cycles between good and bad policies with a period of ~300k–400k
iterations, but each successive peak reaches a higher game value than the last,
indicating long-term improvement.

---

## 4. Summary

| Aspect | Finding |
|--------|---------|
| **Tabular convergence** | All 4 tabular algorithms reach Nash equilibrium (game value ≈ −1/18) |
| **Deep CFR convergence** | Best checkpoint reaches game value −0.0527 (above Nash), but training oscillates |
| **J/K quality** | Deep CFR learns correct bluff/pass frequencies for Jack and King (≈70% bet, ≈23% bluff) |
| **Q quality** | Deep CFR's Queen play (18% bet) is the main gap vs tabular algorithms |
| **Residual noise** | Deep CFR rarely hits exactly 0% or 100% — small (2–4%) constant exploration present |
| **Training dynamics** | Cyclical: network oscillates between good/bad policies; checkpoint recovery mitigates this |
| **Checkpoint mechanism** | Deep CFR saves the best policy encountered during training, avoiding final-policy degradation |

---

## 5. Files

| Algorithm | Log | Model | Visualizations |
|-----------|-----|-------|----------------|
| CFR | `logs/kuhn_strategy_cfr_1e+07.txt` | `models/kuhn_cfr_1e+07.pkl` | `visualizations/kuhn_cfr_1e+07/` |
| CFR+ | `logs/kuhn_strategy_cfr_plus_1e+07.txt` | `models/kuhn_cfr_plus_1e+07.pkl` | `visualizations/kuhn_cfr_plus_1e+07/` |
| DCFR | `logs/kuhn_strategy_dcfr_1e+07.txt` | `models/kuhn_dcfr_1e+07.pkl` | `visualizations/kuhn_dcfr_1e+07/` |
| PDCFR+ | `logs/kuhn_strategy_pdcfr_plus_1e+07.txt` | `models/kuhn_pdcfr_plus_1e+07.pkl` | `visualizations/kuhn_pdcfr_plus_1e+07/` |
| **Deep CFR** | `logs/kuhn_strategy_deep_cfr_1e+06.txt` | `models/kuhn_deep_cfr_1e+06.pkl` | `visualizations/kuhn_deep_cfr_1e+06/` |

---

*Tabular algorithms trained for 10⁷ iterations; Deep CFR trained for 10⁶
iterations with checkpoint recovery. Theoretical Nash value: −1/18 ≈ −0.05556
(Kuhn, 1950).*
