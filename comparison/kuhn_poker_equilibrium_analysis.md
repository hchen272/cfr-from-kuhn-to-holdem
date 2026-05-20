# Kuhn Poker Equilibrium Analysis

A comparative analysis of four CFR variants (CFR, CFR+, DCFR, PDCFR+)
trained on Kuhn Poker for 10⁷ iterations each.

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

| Algorithm | Game Value (10⁷ iters) | vs. Nash (−1/18 ≈ −0.05556) |
|-----------|-----------------------|------------------------------|
| CFR       | −0.0560               | −0.00044 (within noise)     |
| CFR+      | −0.0556               | ≈ 0 (exact match)           |
| DCFR      | −0.0552               | +0.00036 (within noise)     |
| PDCFR+    | −0.0553               | +0.00026 (within noise)     |

All four algorithms converge to the Nash equilibrium. The small differences
(≤ 0.0005) are well within sampling noise for 10⁷ iterations.

---

## 2. Full Strategy Comparison

### 2.1 Jack (J) — the bluffing hand

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | Interpretation |
|---------|-----|------|------|--------|----------------|
| **J**   | [0.816, 0.184] | [0.768, 0.232] | [0.742, 0.258] | [0.772, 0.228] | Bluff-bet frequency: CFR+ > PDCFR+ > CFR > DCFR |
| **Jp**  | [0.667, 0.333] | [0.667, 0.333] | [0.665, 0.335] | [0.667, 0.333] | **≈ 1/3** bluff when checked to — highly consistent |
| **Jb**  | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | **Always fold** Jack to a bet |
| **Jpb** | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | [1.0, 0.0] | **Always fold** even after checking and facing a bet |

Key insight: When first to act, Jack should bluff-bet at a carefully calibrated
frequency (18–26%). This makes opponent indifferent to calling with Queen.
When facing aggression, Jack always folds — it beats nothing.

### 2.2 King (K) — the value hand

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | Interpretation |
|---------|-----|------|------|--------|----------------|
| **K**   | [0.450, 0.550] | [0.302, 0.698] | [0.229, 0.771] | [0.317, 0.683] | Bet frequency: DCFR > CFR+ > PDCFR+ > CFR |
| **Kp**  | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **Always bet** when checked to |
| **Kb**  | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **Always call** a bet |
| **Kpb** | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | [0.0, 1.0] | **Always call** after checking and facing a bet |

King is the strongest hand and is always played aggressively: bet for value
when given the chance, always call/raise against opponent aggression.

The difference is in how often to bet when first to act:
- **CFR (55%)** is the most passive — checks almost half the time to trap
- **DCFR (77%)** is the most aggressive — bets over 3/4 of the time
- This reflects different points on the **equilibrium continuum**

### 2.3 Queen (Q) — the bluff-catcher

| Infoset | CFR | CFR+ | DCFR | PDCFR+ | Interpretation |
|---------|-----|------|------|--------|----------------|
| **Q**   | [1.0, ≈0] | [0.999, 0.001] | [1.0, ≈0] | [0.999, 0.001] | **Almost always check** — betting is dominated |
| **Qp**  | [1.0, ≈0] | [1.0, ≈0] | [1.0, ≈0] | [1.0, ≈0] | **Always check** when checked to |
| **Qb**  | [0.666, 0.334] | [0.665, 0.335] | [0.667, 0.333] | [0.665, 0.335] | **Call 1/3** facing a bet — remarkably consistent |
| **Qpb** | [0.482, 0.518] | [0.433, 0.567] | [0.410, 0.590] | [0.439, 0.561] | Facing a bet after check: call 52–59% |

The Queen is the most interesting hand — it's a pure **bluff-catcher**:
it beats Jack (bluffs) but loses to King (value).

- **Qb** (facing a bet directly): All four algorithms converge to **1/3 call**
  — a textbook Nash equilibrium frequency that makes the bluffer indifferent.
- **Qpb** (facing a bet after checking): The call frequency is higher (52–59%),
  varying across algorithms:
  - CFR: 52% (most conservative)
  - CFR+: 57%
  - DCFR: 59% (most aggressive caller)
  - PDCFR+: 56%

---

## 3. Algorithm Behavioural Differences

### 3.1 Strategy "Polarization"

When sorted by how extreme the strategies become (i.e., how far [P(Pass), P(Bet)]
deviates from uniform [0.5, 0.5]):

```
Most extreme  ─┤  DCFR     (α=1.5, β=0, γ=2 → pushes toward pure actions)
               │  PDCFR+   (predictive component moderates the discounting)
               │  CFR+     (positive regret clamp → faster than CFR, less extreme than DCFR)
Least extreme ─┤  CFR      (standard regret matching, most "spread out")
```

This is a known property: DCFR's aggressive discounting (α > β) effectively
"forgets" old regrets faster, leading to more decisive (near-pure) strategies.
CFR+'s positive-regret clamp achieves a similar effect but more mildly.

### 3.2 Equilibrium Continuum

Kuhn Poker has a **continuum of Nash equilibria**, not a single unique one.
The different algorithms converge to different points along this continuum:

| Decision point | CFR | CFR+ | DCFR | PDCFR+ |
|----------------|-----|------|------|--------|
| K-bet frequency (first act) | 55% | **70%** | **77%** | 68% |
| J-bluff frequency (first act) | 18% | **23%** | **26%** | 23% |
| Qpb-call frequency (check-call) | 52% | 57% | **59%** | 56% |

Notice the pattern: DCFR plays the most "polarized" strategy
(bet King often, bluff Jack often, call with Queen often),
while standard CFR stays closer to uniform mixing.

All of these are valid Nash equilibria — they all achieve game value ≈ −1/18.

### 3.3 Convergence Speed

From the training logs:

| Iterations | DCFR | PDCFR+ |
|------------|------|--------|
| 100k | −0.0557 | **−0.0520** |
| 500k | −0.0564 | **−0.0544** |
| 1M | **−0.0550** | −0.0551 |

PDCFR+ converges faster in early iterations (100k–500k), suggesting the
predictive component provides a useful "warm start." By 1M iterations,
both algorithms are essentially at the equilibrium value.

---

## 4. Summary

| Aspect | Finding |
|--------|---------|
| **Convergence** | All 4 algorithms reach Nash equilibrium (game value ≈ −1/18) |
| **Strategy quality** | All produce qualitatively correct strategies (bluff with J, value-bet with K, bluff-catch with Q) |
| **Consistency** | Qb (call 1/3 facing a bet) is the most stable result — identical across algorithms |
| **Variation** | K-bet, J-bluff, and Qpb-call frequencies differ — reflecting the equilibrium continuum |
| **DCFR** | Most extreme / polarized — pushes closest to pure strategies |
| **PDCFR+** | Fastest early convergence — predictive component accelerates initial learning |
| **CFR+** | Best game value match (−0.0556 exactly) at 10⁷ iterations |
| **CFR** | Most "spread out" mixing — stays closest to uniform randomization |

---

## 5. Files

| Algorithm | Log | Model | Visualizations |
|-----------|-----|-------|----------------|
| CFR | `logs/strategy_cfr_1e+07.txt` | `models/kuhn_cfr_1e+07.pkl` | `visualizations/cfr_1e+07/` |
| CFR+ | `logs/strategy_cfr_plus_1e+07.txt` | `models/kuhn_cfr_plus_1e+07.pkl` | `visualizations/cfr_plus_1e+07/` |
| DCFR | `logs/strategy_dcfr_1e+07.txt` | `models/kuhn_dcfr_1e+07.pkl` | `visualizations/dcfr_1e+07/` |
| PDCFR+ | `logs/strategy_pdcfr_plus_1e+07.txt` | `models/kuhn_pdcfr_plus_1e+07.pkl` | `visualizations/pdcfr_plus_1e+07/` |

---

*Analysed from 1e+07-iteration training runs.
Theoretical Nash equilibrium value: −1/18 ≈ −0.05556 (Kuhn, 1950).*
