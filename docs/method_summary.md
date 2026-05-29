# FANS-MoE 论文方法总结(供迭代)

> 状态:算法层实验进行中,PPL 评估正在修复。本文件总结当前方法定调与证据状态,供进一步迭代论文叙事。
> 目标会议:ICCD(硬件/体系结构倾向),投稿 deadline 2026-05-29。
> 因此叙事必须是 **algorithm-hardware co-design**,而非纯算法压缩。

---

## 1. 问题与立论

MoE(Mixture-of-Experts)LLM 的核心开销是大量 expert 权重的存储与访存。现有压缩工作有两条主线:

- **Expert 粒度**:D²-MoE、Trilemma 等,把 expert 整体做二元/低秩分解。粒度太粗,无法刻画 expert 内部的细结构。
- **量化粒度**:MC-MoE、QMoE 等,对所有权重做均匀或混合精度量化。忽视了 expert 之间的功能冗余。

**我们的立论起点(已被实验确立)**:权重相似 ≠ 功能相似。用权重的 cosine similarity 度量 neuron 冗余是错的——两个 neuron 权重很近不代表功能等价,反之亦然。因此压缩必须建立在 **neuron 的功能行为**度量上,而不是权重几何。

**核心洞察**:在 neuron 粒度上,同一层不同 expert 之间存在一个**连续的功能冗余谱**——有的 neuron 功能在所有 expert 间高度一致(可共享一份),有的只在部分 expert 间一致(可分组共享),有的高度专属(必须独立保留)。这个三档结构是非均匀的,且随网络深度变化。

---

## 2. 方法:FANS-MoE(Functionally-Aligned Neuron-tier Sharing)

一句话:**在 neuron 粒度上,用功能行为(而非权重几何)度量 expert 间冗余,把 neuron 分成三档异构压缩,并用定制 dataflow + 混合精度兑现存储/带宽收益。**

四步流水:

### Step 1 — 功能签名提取(Functional Signature)
用真实文本的 hidden states 做校准,bypass router 强制每个 token 经过全部 routed expert,记录每个 neuron 的激活输出 `SiLU(W_gate·x) ⊙ (W_up·x)` 作为其功能签名。度量功能行为,而非权重。

### Step 2 — 跨 expert neuron 对齐(Permutation Alignment)
同层不同 expert 的 neuron index 因 permutation symmetry 不可直接比较。用 Hungarian/greedy 在功能签名空间上求最优对齐,把各 expert 的 neuron 映射到可比的 slot。

### Step 3 — 功能分档(Tier Allocation)
对每个对齐后的 slot,计算其在 64 个 expert 上的 cross-expert dispersion D_j。数据驱动地(1D 聚类找切点,非固定分位)分成三档:
- **Universal**(D 最低):跨所有 expert 共享一份
- **Group**(D 中等):组内(G 组)共享
- **Specialist**(D 最高):每 expert 独立保留

切点同时受 storage budget 约束(见 §4 公式)。

### Step 4 — 异构压缩 + 重构(Heterogeneous Compression)
- Universal:64 expert 对齐 neuron 加权平均,存 1 份 **BF16**
- Group:KMeans(K=G)取簇心,存 G 份 **FP8**
- Specialist:每 expert 独立 **INT4** 量化
- 前向:三档分别计算后相加;**Universal 部分跨 expert 只算一次**——这是连接硬件 dataflow 的核心红利点

---

## 3. 关键叙事定调(重要,迭代时务必保持)

实验显示功能共享信号**真实但中等**(cross-expert alignment ~0.3,universal 档约占 13–17%)。因此:

**不要写成"大量 universal 共享省掉巨量计算"的纯算法故事——数字撑不起。**

**正确叙事**:MoE 内部存在一个一致的、可功能对齐的三档冗余谱;即便 universal 档只占 ~15%,配合**定制 dataflow(三档对应三级存储层级)+ 混合精度 PE**,仍能带来可观的存储/带宽收益。

这个定调把 sell point 从"算法省得多"转移到"算法发现结构 + 硬件兑现收益",正是投 ICCD(而非 ML 会议)的根本逻辑。**弱信号在 co-design 叙事里反而成了"为什么需要定制硬件"的理由**,逻辑自洽。

---

## 4. 硬件 co-design(ICCD 的核心贡献,paper 后半部分)

三档结构天然对应存储层级与精度层级的异构,这是 reuse 分析的数学结果而非设计选择:

| Tier | Reuse factor | 存储层级 | 精度 | Dataflow |
|---|---|---|---|---|
| Universal | T×E(最高,跨所有 expert/token) | SRAM 常驻 | BF16 | weight-stationary |
| Group | T×E/G(中) | L2 buffer | FP8 | weight-stationary per group |
| Specialist | T×k/E(低,稀疏) | HBM 流式 | INT4 | output-stationary |

**Storage 公式**(per neuron-slot,跨 64 expert):
```
ratio(u,g,s) = (2u + 4g + 32s) / 128
全 universal ≈ 0.016(64×) / 全 group(G=4)=0.031 / 全 specialist=0.25(4×)
有意义的 budget 区间:0.05–0.25
```

**硬件实现**:一个支持 FP16/FP8/INT4 时分复用的 mixed-precision MAC 单元(RTL,综合到开源 PDK 报 area/power/freq),再用 analytical model 外推 N-PE 阵列的 system-level throughput / energy-per-token / bandwidth utilization。

**Dataflow 命名**:Spectrum-Stationary Dataflow(三档对应三级 reuse-stationary 映射)。
**Pipeline**:三档错峰 3-stage 流水(Universal 片上→Group L2→Specialist HBM 流式),含 bubble 分析。

---

## 5. 当前证据状态(诚实清单)

**已确立**:
- 权重相似 ≠ 功能相似(motivation 实验)
- 跨 expert neuron 功能可对齐(real-text 校准后 alignment 0.16→0.26–0.34,方向确立、强度中等)
- 三档结构存在且随层非均匀(D mean 0.78–0.85,数据驱动分档使四层 universal 数 184–242 各异)
- Storage 公式与 budget 机制正确(ratio 精确落在预算上)

**待确立(阻塞项)**:
- **压缩后 PPL 质量**——评估 harness 当前损坏(dense baseline PPL ~13万 = 噪声水平),正在修;修好前无任何可信质量数字
- 三档 vs 二档、功能距离 vs 权重距离的 ablation——未做
- universal 是否对应 DeepSeek 显式的 shared expert(intro 钩子)——未做

**实验配置**:DeepSeek-V2-Lite,代表层 1/5/13/26,64 expert × 1408 neuron,budget 0.08/0.12/0.18。

---

## 6. 与相关工作的差异化(intro 第一段要钉死)

> "neuron 粒度 + 功能对齐的三档异构冗余谱 + 与之 co-design 的 dataflow 和混合精度 PE"

- vs D²-MoE / Trilemma:它们 expert 粒度二元;我们 neuron 粒度三档,刻画 expert 内部结构
- vs MC-MoE / QMoE:它们均匀/混合量化忽视 expert 间冗余;我们按功能冗余分档共享
- vs UbiMoE / MoNDE / Edge-MoE:它们是 MoE 加速器但不利用 neuron 级功能冗余;我们的 dataflow 直接由三档 reuse 结构推导
- vs DeepSeek 的 shared expert:它在 expert 粒度显式设计共享专家;我们将其精细化到 neuron 粒度,并发现被忽视的中间 group 层

---

## 7. 给迭代者的提示

迭代时请保持:
1. **co-design 叙事**不要退回纯算法压缩故事
2. **诚实的信号强度**(~15% universal、alignment ~0.3),不要夸大
3. 硬件部分(dataflow + RTL + analytical 外推)是 ICCD 的录用关键,算法部分是"发现结构",硬件部分是"兑现收益"
4. PPL 主结果表 + uniform INT4 对照 + 三档/二档 ablation 是三个 must-have 实验,目前仅算法流程跑通,数字待修复后填入
