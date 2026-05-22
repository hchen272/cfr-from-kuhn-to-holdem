# Next Steps — Deep CFR 问题分析与改进方向

> 基于当前 Leduc Deep CFR 训练日志 (`leduc_strategy_deep_cfr_1e+06.txt`) 观测与论文对照分析。

---

## 1. 日志观测：Game Value 趋势分析

当前训练配置：`deep_cfr` + Leduc + `--alternate`，1,000,000 episodes。截至 240,000 snapshot 的 game value 序列：

```
Iter     Game Value
10K  →   +0.075     ← 几乎随机
30K  →   +0.396     
50K  →   +0.482     ← 峰值，之后开始回落
70K  →   +0.445
100K →   +0.358
150K →   +0.252
200K →   +0.222
230K →   +0.204
240K →   +0.203     ← 持续下降中
```

### 结论：**不是 monotonically up**

值的走势分为两阶段：
- **上升期** (0–50K)：初始随机策略 → 网络学到有偏策略，P0 获得优势，游戏值被推高
- **下降期** (50K–240K)：随着 P1 也逐渐学习，双方互相调整，游戏值从 +0.48 缓慢下降到 +0.20，趋近 Nash 值 (−0.085)

这与 Deep CFR 的典型行为一致：振荡收敛，而非单调。对比 tabular CFR+ 在 5M iterations 仍为 +0.63，Deep CFR 在 240K episodes 达到 +0.20 已经不算差。**这不是 alternate 模式的 bug。**

---

## 2. Alternate 模式检查

### 2.1 实现逻辑

```python
# train.py — training loop
up = episode % 2 if alternate else -1   # 0 → P0, 1 → P1
episode_util = agent.traverse(cards, "", 1.0, 1.0, update_player=up)
```

```python
# deep_cfr.py — traverse()
player = plays % 2                         # 从 history 长度推断当前玩家
do_accum = (update_player == -1 or update_player == player)

if do_accum:
    strategy_sum[infoset] += reach_prob * strategy   # 累积策略
    buffer.add(features, regret_vec)                  # 存 regret 样本
```

### 2.2 语义分析

| update_player | P0 节点 | P1 节点 |
|---|---|---|
| −1 (both) | 累积策略 + 存 regrets | 累积策略 + 存 regrets |
| 0 | 累积策略 + 存 regrets | **跳过** |
| 1 | **跳过** | 累积策略 + 存 regrets |

- **策略计算**（遍历方向）：始终通过 RegretNet 预测 → regret matching 产生策略，不受 `update_player` 影响
- **策略累积**：只对当前 update_player 的节点累积到 `strategy_sum`
- **Buffer 存储**：只存当前 update_player 的节点产生的瞬时 regrets

### 2.3 Bug 判定：**无明显 bug**

- 遍历树结构完整 — 两个玩家的策略都在每一轮被计算和使用（通过网络），只有一个玩家的数据被存储
- Buffer 随时间会包含均匀的 P0/P1 数据（奇数 episode 存 P1，偶数 episode 存 P0）
- `strategy_sum` dict 按 infoset（含 player 区分）key 存储，不会互相覆盖
- 与 tabular CFR 的 alternation 语义一致

**唯一的微妙之处**：Deep CFR 中网络是根据 buffer 数据训练来预测 regrets 的。当 alternate 时，每轮只有一个玩家的 regrets 入库，Buffer 的 **瞬时分布** 会偏向当前玩家。但长期来看是均衡的，且 reservoir sampling 会保留历史样本，不会被短期偏向污染。

---

## 3. 与论文 Deep CFR 的对比

参考论文：Noam Brown et al., "Deep Counterfactual Regret Minimization", ICML 2019
（已归档至 `references/Deep Counterfactual Regret Minimization.md`）

### 3.1 实现差异总览

| 维度 | 论文 Deep CFR | 当前实现 | 偏差影响 |
|------|--------------|---------|---------|
| **网络数量** | 2 个：Advantage Net + Strategy Net | 1 个：RegretNet | ⚠️ 大 |
| **策略存储方式** | Strategy Net（神经网络）| strategy_sum dict（类似 tabular） | ⚠️ 中 |
| **采样方式** | External Sampling MCCFR | 完整树遍历 | 🟢 小（Leduc 规模不大） |
| **训练策略** | 每 CFR iteration 从零重新训练 | 增量训练（每 TRAIN_FREQ=20 轮训 TRAIN_STEPS=10 步） | 🔴 大 |
| **Loss 加权** | MSE 按 iteration t 加权（Linear CFR） | 无加权 | 🟡 中 |
| **Regret fallback** | 所有 regret ≤ 0 时选最大值 | 所有 regret ≤ 0 时 uniform | 🟢 小 |
| **网络规模** | 7 layers, ~99K params, dual-branch | 3 layers (fc1→fc2→fc3), 256 hidden_dim for Leduc | 🟡 中 |
| **输入编码** | Card embedding + Bet history binary/float | 手牌 rank + history 字符串编码 (~20 维 feature) | 🟡 中 |
| **训练频率** | 4000 SGD steps × batch 10,000 per iteration | 10 steps × batch 512 per 20 episodes | 🟡 中 |
| **Gradient clipping** | `‖g‖ ≤ 1` | 无 | 🟢 小 |
| **Buffer 数量** | 2 个（Advantage Memory + Strategy Memory）| 1 个（仅 Regret Memory） | 🔴 大 |

### 3.2 最关键差异详解

#### 🔴 差异 #1：从零重训练 vs 增量训练

**论文做法**：
```
for each CFR iteration t:
    用最新策略遍历游戏 → 收集 buffer 数据
    神经网络的权重完全重置
    用 buffer 数据训练到收敛（4000 SGD steps）
    用训练好的网络进行下次遍历
```

**当前做法**：
```
for each episode:
    用当前网络遍历游戏 → buffer.add() 逐条追加
    每 20 episodes：做 10 步梯度下降（网络权重复用）
```

**影响**：
1. 论文保证网络完全拟合当前 buffer 中的 regret 分布，不存在"旧策略的惯性"
2. 当前的增量训练可能陷入局部最优 → 网络始终有上一轮策略的阴影 → 收敛变慢甚至卡住
3. 论文明确指出 fine-tuning 比 from-scratch 差很多（Section 17: "Why Train From Scratch?"）

#### 🔴 差异 #2：单一网络 vs 双网络

**论文**：Advantage Net（预测 regrets）+ Strategy Net（预测平均策略）。策略网络在训练结束后单独训练。

**当前**：只有 RegretNet，策略通过 dict 累积（`strategy_sum[infoset] += reach_prob * strategy`）。这是 **tabular-style 策略累积 + neural regret prediction** 的混合。

**影响**：
- Strategy dict 对 Leduc 的 ~620 infosets 完全可行，但策略不享受神经网络的泛化能力
- 论文的两网络设计允许策略也泛化到未见过的 infoset

#### 🟡 差异 #3：Linear CFR 加权

**论文**：MSE loss 中每个样本按 iteration t 加权（`L = E[t * Σ(r̃-V)^2]`），对应 Linear CFR 理论。

**当前**：所有样本等权。

**影响**：Linear CFR 在 tabular 中已被证明加速收敛（O(1/T) vs O(1/√T)）。缺失此加权可能导致慢收敛。

#### 🟡 差异 #4：External Sampling MCCFR

**论文**：每个遍历者在自己的节点探索所有行动，在对手节点只采样一个行动。降低方差 + 减少计算量。

**当前**：完整树遍历（对所有 legal actions 都递归）。

**影响**：对 Leduc（~620 infosets）可忽略。但如果未来扩展到更大游戏，必须引入。

---

## 4. 建议下一步行动

### 4.1 短期验证（不改代码）

- [ ] 让当前 `deep_cfr --alternate` Leduc 跑到 1M episodes 看 game value 是否继续下降到 ~0 附近
- [ ] 跑一组 **不 alternate** 的 Deep CFR 做对照实验：
  ```
  python src/trainer.py -a deep_cfr -g leduc -i 500000       # alternate=False (default)
  ```
  对比 game value 收敛速度

### 4.2 中期改进（按优先级）

| 优先级 | 改进项 | 依据 | 预估工作量 |
|--------|--------|------|-----------|
| **P0** | 实现从零重训练（per CFR iteration） | 论文 Section 17 明确指出 fine-tuning 不如 from-scratch | 大（需重构训练循环） |
| **P1** | 添加 Linear CFR 加权（loss 乘 t） | 论文 Section 10；tabular 已验证 O(1/T) 收敛 | 小（改 loss 计算） |
| **P2** | 所有 regret ≤ 0 时选 argmax 而非 uniform | 论文 Section 2.2，提高近似误差下的鲁棒性 | 极小 |
| **P3** | 添加 Gradient Clipping (`‖g‖ ≤ 1`) | 论文 Section 16 | 极小 |
| **P4** | 添加 Strategy Network（第二网络） | 论文 Section 6.2 | 中 |

### 4.3 长期改进

- [ ] **MCCFR External Sampling**：当扩展到更大游戏时必须引入
- [ ] **双分支网络架构**（Card Branch + Bet Branch）：改进输入编码，提升泛化（论文 Section 13）
- [ ] **更大的训练量**：论文 FHP 实验用 40M buffer + 4000 SGD steps/iteration，当前 512 batch×10 steps 差距悬殊

---

## 5. 当前结论

1. **Game value 趋势不是 bug**：从 +0.48 下降到 +0.20，方向正确。初期上升是 Deep CFR 正常的学习曲线。
2. **Alternate 模式没有代码 bug**：`update_player` 的语义贯穿 accumulate 和 buffer 存储，逻辑一致。
3. **与论文的核心差距**：增量训练（非 from-scratch）+ 单网络（非双网络）+ 无 Linear CFR 加权。这三项是收敛速度受限的主要原因。
4. **优先建议**：先跑完当前的 1M episodes 看最终值，同时起一组 non-alternate 对照。如果值仍远高于 Nash，优先实施 P0（from-scratch retraining）。
