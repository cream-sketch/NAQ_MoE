# FANS-MoE 修复与重跑规范(v2 — 接续 pilot512)

> 上一次 pilot(`outputs_pilot512`)成功验证了 Phase 1-4 的代码可在远端 5090 跑通,但**科学结果全部无效**,原因见下。本文件列出必须修复的问题和重跑要求。
>
> **目标:D6 (May 25) 之前拿到「真实校准 + 数据驱动分档 + 至少一个 PPL 数」。**
> **死线优先级:P0 必须做完,P1 强烈建议,P2 有时间才做。**

---

## 0. 上次 pilot 的三个致命问题(必读)

### 问题 A(P0,最严重):用了 `random_hidden` 校准
pilot 的 `Calibration source: random_hidden`。整套方法的根基是「用**真实文本**测量 neuron 功能签名」,随机噪声把语言模型里的共享计算结构整个抹掉了。

**后果**:`D mean ≈ 0.96`(所有 neuron 互相都不相似)。这个数如果当真,等于宣称「不存在 universal neuron」,直接否定 paper 的核心 motivation。这是 random calibration 的 artifact,不是发现。

**判定修复成功的标准**:真实 C4 校准后,D 的直方图应当出现**低 dispersion 的峰 + 高 dispersion 的尾**(双峰或明显长尾),而不是全挤在 0.96。alignment after 应当从 0.16 明显上升。

### 问题 B(P0):三档划分是写死的分位数,没有判别力
pilot 四个层(1/5/13/26)的 tier 分布**完全相同**:U=352 / G=633 / S=423,即固定的 25%/45%/30% 分位切割。这意味着 tier allocation 在任何数据上都吐同样的结果,与 D 矩阵真实形状无关。

**判定修复成功的标准**:改成数据驱动切点后,**不同层的 U/G/S 分布必须不同**(预期浅层 universal 多、深层 specialist 多)。

### 问题 C(P0):Phase 5-6 没跑,没有任何 PPL
压缩和 PPL 重构都是 `not run`。目前**没有一个 perplexity 数字**,没有验证压缩后模型可用。没有 Phase 6 就没有论文主结果表。

### 问题 D(P1):storage formula 有 bug(我方 spec 错误)
旧公式分母和 per-neuron 字节数没对齐,导致 `budget_nonbinding=true`(全 specialist INT4 已是 0.125,预算永不约束)。下面给正确公式。

---

## 1. 修复 A:真实 C4 校准(替换 random_hidden)

### 1.1 数据获取
远端无法通过 `ial-jump` 传数据集,改用 HF 镜像直接流式拉取:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
samples = []
for i, ex in enumerate(ds):
    if i >= 256: break
    samples.append(ex["text"])
```

如果 `allenai/c4` 镜像拉不动,**fallback 顺序**:
1. `wikitext`, `wikitext-103-raw-v1`, split=`train`(小、稳、镜像一定有)
2. `Salesforce/wikitext`
3. 本地任何 `.txt` 语料(英文,>1MB)

**绝对不要再用 `random_hidden`。** 校准数据必须是真实自然语言。

### 1.2 预处理
- tokenize,每条截断到 512 token
- 取 256 条(磁盘紧张则 64 条 × 256 token,见 §1.4)
- 保存 `data/calibration_c4.pt`,记录来源到 metadata(`calibration_source: "c4"` 或 `"wikitext"`)

### 1.3 提取激活(沿用 pilot 的 bypass-router 逻辑)
- 对 layer 1/5/13/26(先这四层,验证后再扩全层)
- 强制每个 token 走全部 64 routed expert + 2 shared expert
- 提取 `SiLU(W_gate@x) * (W_up@x)`,即每个 neuron 的真实激活
- detach → cpu → BF16 存储

### 1.4 显存/磁盘
- pilot 的 358M / 309 文件说明用了精简存储,沿用同样策略
- 若 256×512 存储超限,降到 **64 样本 × 256 token**
- 关键:不在 GPU 累积,每 expert 处理完立刻落盘

### 1.5 必做 sanity check(不通过不进入下一步)
```
1. 打印 D 的直方图(20 bins),确认不再是单峰在 0.96
   —— 期待:有低 dispersion 的一簇 neuron
2. 打印 alignment before/after,after 应当 >> 0.16
   —— 如果 after 仍 ≈0.16,说明真实数据下功能对应也很弱,
      立即报告,不要继续(这是个需要讨论的信号)
3. 抽查一个 universal 候选 neuron(D 最低的几个):
   它在 64 个 expert 上的激活应当高度相关(cos > 0.7)
```

---

## 2. 修复 B:数据驱动的三档切点(替换固定分位)

### 2.1 废弃做法
删掉「25%/45%/30% 固定分位」。tier 切点必须由 D 分布形状决定。

### 2.2 主方法:1D 聚类找切点
对每层的 D 向量(1408 个值)做 **1D 三类聚类**:

```python
from sklearn.cluster import KMeans
import numpy as np

def data_driven_tiers(D, n_tiers=3):
    """D: (1408,) per-slot dispersion. 返回 tier_map (1408,) ∈ {0,1,2}"""
    km = KMeans(n_clusters=n_tiers, n_init=10).fit(D.reshape(-1, 1))
    centers = km.cluster_centers_.flatten()
    # 按 center 升序重映射:最低 dispersion → Universal(0)
    order = np.argsort(centers)          # order[0] = 最低 center 的簇 id
    relabel = {old: new for new, old in enumerate(order)}
    tier_map = np.array([relabel[l] for l in km.labels_], dtype=np.int32)
    # tier 0 = Universal(D最低), 1 = Group, 2 = Specialist(D最高)
    return tier_map, np.sort(centers)
```

### 2.3 辅助方法(可选):GMM + 谷底
如果 KMeans 切点不稳,改用 1D GMM 拟合,取相邻分量之间的密度谷底作为 tau1/tau2。先用 KMeans,简单优先。

### 2.4 预算约束的接法
数据驱动切点先给出「自然分档」,再**检查是否满足 storage 预算**(§3 公式)。若超预算,沿 D 轴把切点往 specialist 方向收紧(把一部分 group 降级)直到满足。记录最终 (u,g,s) 和实际 storage ratio。

### 2.5 必做 sanity check
```
1. 打印四个层(1/5/13/26)各自的 U/G/S 分布
   —— 必须不再完全相同
2. 画一张跨层堆叠柱状图(这是 paper 的 Figure 1)
   —— 期待趋势:浅层 U 比例高,深层 S 比例高
3. 若四层仍高度一致,报告出来 —— 可能是 D 区分度不够,需要讨论 metric
```

---

## 3. 修复 D:正确的 storage formula

替换 spec v1 Phase 4.3 里的旧公式。per neuron-slot(跨 64 expert):

```
原始       = 64 × (3·d_model) × 2 bytes      # 64 份 BF16
Universal  = 1  × (3·d_model) × 2 bytes       # 共享 1 份 BF16
Group(G=4) = 4  × (3·d_model) × 1 byte        # 4 份 FP8
Specialist = 64 × (3·d_model) × 0.5 byte      # 64 份 INT4

ratio(u,g,s) = [u·(1×2) + g·(4×1) + s·(64×0.5)] / (64×2)
             = (2u + 4g + 32s) / 128
```

验证:
- 全 universal:2/128 ≈ 0.016(64× 压缩)
- 全 group(G=4):4/128 = 0.031
- 全 specialist:32/128 = 0.25(4× 压缩)

所以**有意义的 budget 区间是 0.05 ~ 0.25**。论文跑 **0.08 / 0.12 / 0.18** 三个点(替换掉旧的 0.4/0.5/0.6,那些永远不约束)。

> 注:G 是 group 数。先用 G=4。若想压更狠可调 G=2。FP8 当前若 torch 版本不支持 `float8_e4m3fn`,先用 BF16 占位但**字节数仍按 FP8=1 计入 storage**(标注 `fp8_emulated=true`),真实 FP8 内核留给硬件部分。

---

## 4. 修复 C:打通 Phase 5-6,拿到第一个 PPL

### 4.1 压缩算子(Phase 5,沿用 spec v1,三点确认)
- Universal:64 expert 对齐后 neuron 取 **uniform 平均**(router-freq 加权留作 ablation)
- Group:对齐后 neuron 做 **KMeans(K=G=4)** 取簇心;记录 `group_assignment[64]`
- Specialist:**per-tensor 对称 INT4**(per-channel 留作 ablation)

### 4.2 PPL 重构 —— 用回退方案先拿数(关键决策)
**不要在 D6 前死磕 monkey-patch 的 compute reuse。** 先走最稳的路径拿到 PPL:

> **回退路径(P0)**:把每个 expert 的权重「压缩→解压」回 BF16,塞回原模型,跑标准 forward 测 PPL。
> - Universal slot:64 个 expert 的该 neuron 全部替换成那 1 份共享权重(解压回各自位置)
> - Group slot:每个 expert 用它所属 group 的簇心权重替换
> - Specialist slot:INT4 量化再反量化回 BF16
> - 这样**不改 forward 结构**,只是权重变了,直接 `model(input_ids, labels=input_ids)` 测 PPL
> - 失去 compute reuse 的「速度红利」,但**完整保留压缩对质量的影响** —— 这正是 PPL 要测的东西

compute reuse 的真实 forward(monkey-patch `DeepseekV2MoE.forward`)留作 P2,跑通回退路径之后再做。

### 4.3 PPL 评估
```python
# wikitext-2-raw-v1 test, sliding window
max_length, stride = 2048, 1024
# 累积 nll,返回 exp(mean nll)
```

测三组,填进 `outputs/ppl_results.json`:
```json
{
  "dense_baseline": <数>,
  "uniform_int4_allexpert": <数>,
  "fans_moe": {"0.08": <数>, "0.12": <数>, "0.18": <数>}
}
```

### 4.4 必做 sanity check
```
1. dense_baseline PPL 应当在合理范围(DeepSeek-V2-Lite 在 wikitext-2 约 6-8)
   —— 如果 baseline 就 >20,说明加载/评估有 bug,先修这个
2. fans_moe 各档 PPL 不应 >20(>20 说明压缩或解压有 bug)
3. 压缩率越低(压得越狠)PPL 越高,应当单调 —— 不单调就是 bug
```

---

## 5. 执行顺序与时间预算

```bash
# Step 1 (P0): 真实校准重跑 Phase 1-4,四个代表层
python src/fans_moe_lite.py --config configs/deepseek_v2_lite.yaml \
    --calibration c4 --layers 1,5,13,26 --tokens 512 \
    --tier-method data_driven --output-dir outputs_c4_512 --force
# 跑完看 §1.5 + §2.5 的 sanity check

# Step 2 (P0): Phase 5 压缩 + Phase 6 回退路径 PPL
python src/compress_and_ppl.py --config configs/deepseek_v2_lite.yaml \
    --weights-dir outputs_c4_512 --budgets 0.08,0.12,0.18 \
    --reconstruct fallback_bf16 --ppl-dataset wikitext-2-raw-v1
# 跑完看 §4.4 的 sanity check

# Step 3 (P1): 全 26 层重跑(四层验证 OK 后)
#   仅扩大 --layers 到 1-26

# Step 4 (P2): compute-reuse 真实 forward + ablations
```

**时间预算**:Step 1 真实校准 ~4-6h,Step 2 ~2-3h。一天内能拿到第一个真实 PPL。

---

## 6. 同时要做的两件小事(给 paper 用)

### 6.1 Shared expert 对齐分析(paper intro 钩子)
DeepSeek-V2-Lite 自带 2 个 shared expert。在 Phase 3 之后加 ~15 行:
- 算 routed expert 中划归 universal 的 neuron,其功能签名是否 cluster 到 shared expert 附近
- 如果是,paper 可以写「DeepSeek 在 expert 粒度用 shared expert 验证了通用特征存在,我们精细到 neuron 粒度并发现了被忽视的 group 中间层」
- 输出:`outputs/shared_alignment.json`,记录 universal-vs-shared 的平均相似度

### 6.2 记录所有中间数字
每个 phase 把关键指标写 csv:层、D 分布分位、alignment delta、U/G/S 分布、storage ratio、PPL。paper 的 table 直接从这里取。

---

## 7. 哪些坑别再踩

1. **不要再用 random / 合成校准数据** —— 任何情况下都用真实文本
2. **不要再用固定分位切 tier** —— 必须数据驱动且验证跨层差异
3. **不要为了 compute reuse 卡住整个 Phase 6** —— 先用 fallback_bf16 拿 PPL
4. **不要扩全 26 层之前就跑** —— 先用 4 层验证 sanity check 全过
5. **每个 phase 的 sanity check 不过,不进下一步** —— 带病往下跑会浪费整天

---

## 8. 跑完后回报给我的清单

按这个顺序贴给我,我来判断算法层能不能 freeze:
1. D 直方图(四层)—— 看是否双峰/长尾
2. alignment after 的四层数值 —— 看是否 >> 0.16
3. 四层的 U/G/S 分布 —— 看是否不同
4. `ppl_results.json` —— 看 dense baseline 合理 + fans_moe 没崩
5. shared_alignment.json —— 看 universal 是否贴近 shared expert
