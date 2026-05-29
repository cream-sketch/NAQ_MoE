#MoE 算法层实现规范

> 目标:在 RTX 5090(32GB)上,基于 `deepseek-ai/DeepSeek-V2-Lite`,实现 neuron 粒度三档异构压缩(Universal / Group / Specialist),并验证压缩后 perplexity。
> 论文场景:ICCD 投稿,9 天 deadline,算法层结果必须 D6 (May 25) 之前 lock。

---

## 0. 总体说明

### 0.1 模型参数(DeepSeek-V2-Lite 关键超参)
- `hidden_size` (d_model): 2048
- `moe_intermediate_size` (d_ff per expert): 1408
- `num_hidden_layers`: 27
- `first_k_dense_replace`: 1(**第 0 层是 dense FFN,不参与 MoE 分析,跳过**)
- `n_routed_experts`: 64
- `n_shared_experts`: 2(**shared experts 不参与三档划分,作为参照组单独分析**)
- `num_experts_per_tok` (top-K): 6
- Expert 内部是 SwiGLU 结构:`out = (SiLU(W_gate @ x) * (W_up @ x)) @ W_down`

### 0.2 显存预算策略
- 加载模型用 `torch_dtype=torch.bfloat16`(BF16),全模型约 31.4 GB,加上 activation buffer 可能 OOM
- **校准阶段**:用 `device_map="auto"` 让 transformers 自动把不活跃层 offload 到 CPU,只把当前处理层放在 GPU
- **PPL 评估阶段**:用 BF16 全模型加载,关掉所有其他进程
- 校准 hook 输出立刻 `.detach().cpu()`,绝不在 GPU 上累积

### 0.3 环境
```bash
pip install torch>=2.4 transformers>=4.45 accelerate datasets scipy scikit-learn tqdm
```

### 0.4 项目目录结构(必须按此组织)
```
fans_moe/
├── configs/
│   └── deepseek_v2_lite.yaml         # 模型超参 + 实验配置
├── data/
│   └── calibration_samples.pt        # 校准数据(预处理后)
├── src/
│   ├── extract_activations.py        # Phase 1
│   ├── align_neurons.py              # Phase 2
│   ├── compute_distance.py           # Phase 3
│   ├── tier_allocation.py            # Phase 4
│   ├── compress_tiers.py             # Phase 5
│   ├── forward_reconstruct.py        # Phase 6
│   └── utils.py                      # 公共工具
├── outputs/
│   ├── activations/                  # 每层激活信号(BF16 npy)
│   ├── alignments/                   # 每层的 permutation(int32 npy)
│   ├── distances/                    # 距离矩阵
│   ├── tier_maps/                    # 每层 neuron → tier 的映射
│   ├── compressed_weights/           # 压缩后的权重
│   └── ppl_results.json
└── run_all.sh                        # 一键串起所有 phase
```

---

## Phase 1: 校准激活提取(Activation Signature Extraction)

### 1.1 目标
对每一 MoE 层的每个 routed expert 的每个 neuron(W_up 的每一行 + W_gate 的每一行),提取一个"功能签名"——即该 neuron 在固定校准输入上的输出向量。这个签名是后续对齐和三档划分的基础。

### 1.2 关键设计决策

**Bypass routing.** 为了让不同 expert 的 neuron 在同样的输入上做对比,**校准阶段必须绕过 router**——把每个 token 强制送进**所有 64 个 routed expert**(以及 shared expert)。这是 calibration mode,不是 inference mode。

**只对前向激活做 hook,不存储 attention 等其他模块的中间结果。**

### 1.3 数据
- 从 HuggingFace 的 `allenai/c4` (en, train split) 取 **256 条样本**,每条截断到 **512 tokens**
- 保存为 `data/calibration_samples.pt`,内容是 tokenized input_ids 张量,shape `(256, 512)`
- 用 DeepSeek-V2-Lite 自带的 tokenizer

### 1.4 算法

```python
# 伪代码:每层的处理流程
for layer_idx in range(1, 27):  # 跳过 layer 0(dense FFN)
    moe_layer = model.layers[layer_idx].mlp
    
    # 准备存储:layer 的每个 expert × 每个 neuron 在每个 token 上的激活
    # shape: (num_experts=64, d_ff=1408, num_tokens=256*512=131072)
    # 用 BF16 存储:64 × 1408 × 131072 × 2 bytes ≈ 24 GB —— 太大!
    # 改用:分 expert 分批处理,每个 expert 单独存 .pt 文件
    
    for expert_idx in range(64):
        expert = moe_layer.experts[expert_idx]
        
        signatures = []  # 收集这个 expert 所有 neuron 在所有 token 上的输出
        
        for batch in calibration_batches:
            x = get_hidden_state_before_moe(batch, layer_idx)  # shape (B, T, d_model)
            
            # 强制走这个 expert(bypass router)
            gate_out = x @ expert.gate_proj.weight.T   # (B, T, d_ff)
            up_out = x @ expert.up_proj.weight.T        # (B, T, d_ff)
            neuron_act = F.silu(gate_out) * up_out      # (B, T, d_ff)
            # 注意:这里 neuron_act[:, :, j] 就是 expert e 第 j 个 neuron 在每个 token 上的激活
            
            signatures.append(neuron_act.flatten(0, 1).detach().to(torch.bfloat16).cpu())
            # shape after flatten: (B*T, d_ff)
        
        sig = torch.cat(signatures, dim=0)  # (256*512, 1408)
        # 存为 outputs/activations/layer{layer_idx:02d}_expert{expert_idx:02d}.pt
```

**重要细节**:
- `get_hidden_state_before_moe(batch, layer_idx)` 需要用 hook 拿到该层 MoE 输入(即 attention 之后、router 之前的 hidden state)。用 `register_forward_pre_hook` 在 `model.layers[layer_idx].mlp` 上注册。
- 每次只 forward 一层的 attention 部分(到 MoE 输入)就停下,不要 forward 完整模型——节省计算。
- Shared experts(2 个)也用同样的方式提取签名,存为 `layer{L:02d}_shared{0,1}.pt`。

### 1.5 输出
- 文件:`outputs/activations/layer{L:02d}_expert{E:02d}.pt`,L=1..26, E=0..63
- 每个文件:BF16 tensor of shape `(131072, 1408)`,约 369 MB
- 总存储:26 层 × 64 expert × 369 MB ≈ 600 GB
- **如果磁盘不够,缩减校准数据到 64 样本 × 256 tokens = 16384 tokens**,文件大小降到 46 MB,总量约 75 GB

### 1.6 Sanity check
- 抽查一个 expert 的签名:它的某个 neuron 输出应该不全是 0(否则该 neuron 死了)
- 抽查两个不同 expert 的同 index neuron:它们的输出应该有差别(否则 routing 学得很差或代码错了)

---

## Phase 2: 跨 Expert 的 Neuron 对齐(Permutation Alignment)

### 2.1 目标
同层 64 个 expert 的 neuron index 不直接可比(神经网络的 permutation symmetry)。需要为每个 expert 找一个 permutation,把它的 neuron 排列到与"参考 expert"功能对齐的位置。

### 2.2 算法选择
**主路径:Hungarian(scipy.optimize.linear_sum_assignment)**——最优解,O(d_ff^3),1408^3 ≈ 2.8B ops,scipy 实现约 10-20 秒/对。

**Fallback:Greedy 贪心匹配**——O(d_ff^2 log d_ff),约 2 秒/对。如果 Hungarian 超时再切换。

### 2.3 参考 expert 的选择
- 用 expert 0 作为参考(最简单)
- 或者:选 router 统计中激活频率最高的 expert(更鲁棒,但需要先跑一遍统计)
- **先用 expert 0,简单优先**

### 2.4 算法

```python
from scipy.optimize import linear_sum_assignment

def compute_alignment_cost(sig_ref, sig_target):
    """
    sig_ref: (N_tokens, d_ff) — 参考 expert 的所有 neuron 在所有 token 上的输出
    sig_target: (N_tokens, d_ff) — 待对齐 expert
    返回:cost matrix C,shape (d_ff, d_ff)
    C[i, j] = -cosine_similarity(sig_ref[:, i], sig_target[:, j])
    """
    # Normalize columns to unit norm
    R = sig_ref.float() / (sig_ref.float().norm(dim=0, keepdim=True) + 1e-8)
    T = sig_target.float() / (sig_target.float().norm(dim=0, keepdim=True) + 1e-8)
    
    # Cosine similarity matrix
    sim = R.T @ T  # (d_ff, d_ff)
    
    # Convert to cost (negate)
    cost = -sim.numpy()
    return cost

def hungarian_align(layer_idx):
    sig_ref = torch.load(f"outputs/activations/layer{layer_idx:02d}_expert00.pt")
    
    permutations = {}
    for e in range(64):
        if e == 0:
            permutations[e] = np.arange(1408)  # 参考 expert,identity
            continue
        sig_e = torch.load(f"outputs/activations/layer{layer_idx:02d}_expert{e:02d}.pt")
        cost = compute_alignment_cost(sig_ref, sig_e)
        row_ind, col_ind = linear_sum_assignment(cost)
        # col_ind[i] 表示 expert e 的第 col_ind[i] 个 neuron 对齐到 expert 0 的第 i 个
        permutations[e] = col_ind
    
    np.save(f"outputs/alignments/layer{layer_idx:02d}_perms.npy", permutations)
```

### 2.5 输出
- `outputs/alignments/layer{L:02d}_perms.npy`:dict of `{expert_idx: np.array(d_ff,)}`
- 26 个文件,每个几 KB

### 2.6 Sanity check
- 检查每个 permutation 都是合法置换(每个 index 出现且仅出现一次)
- 检查对齐质量:对齐后的同 index neuron,其签名相关性应当显著高于对齐前

```python
# 验证 alignment 有效
sig_0 = load_sig(layer=5, expert=0)              # (N, d_ff)
sig_1_raw = load_sig(layer=5, expert=1)
perm_1 = perms[1]
sig_1_aligned = sig_1_raw[:, perm_1]              # 应用 permutation

# 对角线 cosine 平均值,对齐后应当显著高于未对齐
corr_unaligned = (sig_0 * sig_1_raw).sum(0) / (sig_0.norm(0) * sig_1_raw.norm(0))
corr_aligned = (sig_0 * sig_1_aligned).sum(0) / (sig_0.norm(0) * sig_1_aligned.norm(0))
print(f"unaligned mean corr: {corr_unaligned.mean()}")
print(f"aligned mean corr:   {corr_aligned.mean()}")  # 应当明显高
```

---

## Phase 3: 功能距离矩阵(Functional Distance Matrix)

### 3.1 目标
对每一层、每个对齐后的 neuron slot j(j = 0..1407),计算该 slot 在 64 个 expert 上的 functional 多样性度量。这个度量将驱动三档划分。

### 3.2 度量公式

对 slot j,有 64 个对齐后的 neuron 签名 `{s_e^j}_{e=0..63}`,每个 shape `(N_tokens,)`。

**Per-slot dispersion(主度量)**:
```
D_j = mean over expert pairs (e1, e2) of (1 - cos_sim(s_{e1}^j, s_{e2}^j))
```
- D_j 接近 0 → 所有 expert 在这个 slot 上功能几乎一致 → Universal 候选
- D_j 中等 → 部分 expert 一致 → Group 候选
- D_j 接近 1 → 几乎全部 expert 都不一致 → Specialist 候选

**Per-slot cluster structure(辅助度量,用于 Group 划分)**:
对 slot j 的 64 个签名做 KMeans(K=4 或 8),记录:
- 最大簇的 expert 数(`max_cluster_size`)
- 簇数和簇间距离

### 3.3 算法

```python
def compute_distance_matrix(layer_idx, perms):
    sigs = []  # 64 个对齐后的签名,shape (N_tokens, d_ff) each
    for e in range(64):
        sig_e = torch.load(f"outputs/activations/layer{layer_idx:02d}_expert{e:02d}.pt").float()
        sig_e = sig_e[:, perms[e]]  # 应用 alignment
        sig_e = sig_e / (sig_e.norm(dim=0, keepdim=True) + 1e-8)  # column-normalize
        sigs.append(sig_e)
    
    # Stack: shape (64, N_tokens, d_ff)
    sigs = torch.stack(sigs, dim=0)  # 注意:可能很大,考虑分 slot 处理
    
    D = torch.zeros(1408)
    cluster_info = []
    for j in range(1408):
        # slot j 在 64 个 expert 上的签名: (64, N_tokens)
        slot_sigs = sigs[:, :, j]  # (64, N_tokens)
        
        # pairwise cosine,因为已归一化,直接 inner product
        sim_matrix = slot_sigs @ slot_sigs.T  # (64, 64)
        # 取上三角(不含对角)
        mask = torch.triu(torch.ones(64, 64), diagonal=1).bool()
        avg_sim = sim_matrix[mask].mean()
        D[j] = 1.0 - avg_sim
        
        # KMeans for cluster structure
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=4, n_init=3).fit(slot_sigs.numpy())
        max_cluster = max(np.bincount(km.labels_))
        cluster_info.append({
            'max_cluster_size': max_cluster,
            'inertia': km.inertia_,
        })
    
    np.save(f"outputs/distances/layer{layer_idx:02d}_D.npy", D.numpy())
    json.dump(cluster_info, open(f"outputs/distances/layer{layer_idx:02d}_clusters.json", "w"))
```

### 3.4 输出
- `outputs/distances/layer{L:02d}_D.npy`:shape `(1408,)`,float32
- `outputs/distances/layer{L:02d}_clusters.json`:每个 slot 的簇结构信息

### 3.5 Sanity check
- D 的分布应该是连续的,从接近 0 到接近 1
- 画直方图,目视检查是否有"双峰"或"长尾"结构(支持三档划分的存在性)

---

## Phase 4: 三档分配算法(Tier Allocation)

### 4.1 目标
基于 D_j(以及 cluster_info),把每个 slot 划分到 Universal / Group / Specialist。

### 4.2 划分规则(主算法:阈值切 + cluster 结构判断)

```python
def allocate_tiers(D, cluster_info, tau1, tau2, max_cluster_threshold=32):
    """
    D: (1408,) per-slot dispersion
    tau1, tau2: 阈值
    max_cluster_threshold: 簇覆盖多少 expert 算"主导簇"
    
    返回:tier_map (1408,),取值 {0: Universal, 1: Group, 2: Specialist}
    """
    tier_map = np.zeros(1408, dtype=np.int32)
    for j in range(1408):
        if D[j] < tau1 and cluster_info[j]['max_cluster_size'] >= max_cluster_threshold:
            tier_map[j] = 0  # Universal
        elif D[j] < tau2:
            tier_map[j] = 1  # Group
        else:
            tier_map[j] = 2  # Specialist
    return tier_map
```

### 4.3 阈值的预算反推(关键创新点)

给定目标压缩率 `B`(论文里跑 40%、50%、60% 三个点),求解最优 `(tau1, tau2)`。

**存储模型**:
设 universal 占比 u%、group 占比 g%、specialist 占比 s%(u+g+s=1)。每个 neuron 占 `d_model + 1`(W_up 一行 + W_gate 一行 + W_down 一列)= `2 * d_model + d_ff` 实际是 `3 * d_model` 个参数(忽略 d_ff 端,因为每个 neuron 对应 W_up 一行 + W_gate 一行 + W_down 一列,各 d_model 长度)。

每个 neuron 的"原始"存储 = `3 × d_model × 2 bytes`(BF16)= `12 × d_model` bytes

每个 neuron 在压缩后的存储:
- Universal:**只存 1 份,所有 64 expert 共享**,精度 BF16 → `1 × 3 × d_model × 2 = 6 × d_model` bytes
- Group:**存 G 份**(G=4 或 8,先用 G=4),精度 FP8 → `G × 3 × d_model × 1 = 3G × d_model` bytes
- Specialist:**每 expert 独立一份**,精度 INT4(0.5 byte/param)→ `64 × 3 × d_model × 0.5 = 96 × d_model` bytes

总原始(每个 neuron):`64 × 12 × d_model = 768 × d_model` bytes

总压缩(每个 neuron,加权):
```
storage(u, g, s) = u × 6 × d_model 
                + g × 3G × d_model 
                + s × 96 × d_model
                = d_model × (6u + 3G·g + 96s)
```

压缩率 = `storage / 768 × d_model` = `(6u + 3G·g + 96s) / 768`

对 G=4:`= (6u + 12g + 96s) / 768`

**反推方程**:给定目标压缩率 B,求 (u, g, s) 满足 u+g+s=1 且 (6u+12g+96s)/768 ≤ B 且最小化 PPL(代理目标:最大化保留的总信息,即最小化 sum(D_j for j in specialist) + ...)。

简化:线性扫描 (tau1, tau2),计算每个组合下的 (u, g, s),挑满足预算且 dispersion-weighted 损失最小的组合。

```python
def find_best_thresholds(D, cluster_info, target_compression, G=4, d_model=2048):
    candidates = []
    tau1_grid = np.linspace(0.05, 0.4, 20)
    tau2_grid = np.linspace(0.3, 0.8, 20)
    
    for tau1 in tau1_grid:
        for tau2 in tau2_grid:
            if tau1 >= tau2: continue
            tier_map = allocate_tiers(D, cluster_info, tau1, tau2)
            u = (tier_map == 0).mean()
            g = (tier_map == 1).mean()
            s = (tier_map == 2).mean()
            compression = (6*u + 3*G*g + 96*s) / 768
            if compression > target_compression: continue
            
            # 代理损失:specialist 的 D 之和应该尽可能大(高 D 的 neuron 才放 specialist)
            # universal 的 D 之和应该尽可能小(低 D 的 neuron 才放 universal)
            loss = D[tier_map == 0].sum() - D[tier_map == 2].sum()
            candidates.append((loss, tau1, tau2, u, g, s, compression))
    
    candidates.sort()
    return candidates[0]  # 损失最小的组合
```

### 4.4 输出
- `outputs/tier_maps/layer{L:02d}_tiers_B{compression}.npy`:shape `(1408,)`,int32
- `outputs/tier_maps/layer{L:02d}_meta_B{compression}.json`:`{tau1, tau2, u, g, s, actual_compression}`

### 4.5 Sanity check
- 输出 tier 分布:典型预期是 u ∈ [0.1, 0.3], g ∈ [0.3, 0.6], s ∈ [0.2, 0.5](具体看层)
- 浅层应该 universal 比例更高,深层更多 specialist(可视化时画一个跨层的堆叠柱状图)

---

## Phase 5: 每档的压缩/精度方案

### 5.1 Universal 合并

**算法:加权平均**

```python
def merge_universal(layer_idx, slot_j, perms, weight_type='uniform'):
    """
    universal slot 在 64 expert 上合并成 1 份。
    weight_type: 'uniform' (简单平均) 或 'router_freq' (按 router 激活频率加权)
    """
    moe_layer = model.layers[layer_idx].mlp
    
    W_up_stack = []   # (64, d_model)
    W_gate_stack = []
    W_down_stack = [] # (64, d_model)
    
    for e in range(64):
        expert = moe_layer.experts[e]
        # 原始 W_up shape: (d_ff, d_model),取第 perms[e][slot_j] 行
        # 因为 slot_j 是对齐后的 index,我们要找 expert e 中对齐到 slot_j 的原始 index
        # perms[e][i] = (expert e 的哪个 neuron 对齐到参考 expert 的第 i 个 slot)
        # 但 phase 2 里我们用的是 col_ind[i],表示 expert e 的第 col_ind[i] 个对齐到 ref 的 i
        # 所以 expert e 中对应 slot_j 的原始 index 是 perms[e][slot_j]
        orig_idx = perms[e][slot_j]
        W_up_stack.append(expert.up_proj.weight[orig_idx])     # (d_model,)
        W_gate_stack.append(expert.gate_proj.weight[orig_idx])
        W_down_stack.append(expert.down_proj.weight[:, orig_idx])  # (d_model,)
    
    W_up_stack = torch.stack(W_up_stack)  # (64, d_model)
    # ...
    
    if weight_type == 'uniform':
        w = torch.ones(64) / 64
    elif weight_type == 'router_freq':
        w = router_freq[layer_idx]  # 预先统计的 (64,)
        w = w / w.sum()
    
    W_up_universal = (W_up_stack * w[:, None]).sum(0)        # (d_model,)
    W_gate_universal = (W_gate_stack * w[:, None]).sum(0)
    W_down_universal = (W_down_stack * w[:, None]).sum(0)
    
    return W_up_universal.to(torch.bfloat16), W_gate_universal.to(torch.bfloat16), W_down_universal.to(torch.bfloat16)
```

**默认用 uniform。**Router-freq 作为 ablation。

### 5.2 Group 共享

**算法:KMeans 簇心**

每个 group slot j,把 64 expert 的对齐后 neuron 做 KMeans(K=G=4),取每个簇的簇心作为该 group 的共享权重。

```python
def share_group(layer_idx, slot_j, perms, G=4):
    # 收集 64 expert 的对齐后 W_up 行
    W_up_stack = []  # 同上
    W_gate_stack = []
    W_down_stack = []
    for e in range(64):
        orig_idx = perms[e][slot_j]
        W_up_stack.append(expert.up_proj.weight[orig_idx])
        # ...
    
    W_up_stack = torch.stack(W_up_stack).float()  # (64, d_model)
    
    # KMeans 在拼起来的特征空间上做(W_up + W_gate + W_down 拼起来)
    features = torch.cat([
        torch.stack(W_up_stack),
        torch.stack(W_gate_stack),
        torch.stack(W_down_stack),
    ], dim=1).float()  # (64, 3*d_model)
    
    km = KMeans(n_clusters=G, n_init=5).fit(features.numpy())
    
    # 簇心 → group 共享权重
    centroids = torch.tensor(km.cluster_centers_)  # (G, 3*d_model)
    
    # 拆回 3 份
    W_up_groups = centroids[:, :d_model]
    W_gate_groups = centroids[:, d_model:2*d_model]
    W_down_groups = centroids[:, 2*d_model:]
    
    # group_assignment[e] = expert e 属于哪个 group
    group_assignment = km.labels_  # (64,)
    
    # 用 FP8 量化(或 BF16,取决于硬件)
    # 简化:先用 BF16,FP8 用 torch.float8_e4m3fn (需要 torch >=2.4)
    W_up_groups_fp8 = W_up_groups.to(torch.bfloat16)  # 暂用 BF16
    # ...
    
    return W_up_groups_fp8, W_gate_groups_fp8, W_down_groups_fp8, group_assignment
```

### 5.3 Specialist 量化

**算法:Per-channel symmetric INT4**

每个 specialist slot,**每 expert 独立保留**,但用 INT4 量化。

```python
def quantize_int4_symmetric(W):
    """
    W: tensor of shape (d,)
    返回 W_int4 (int8 packing) 和 scale
    """
    abs_max = W.abs().max()
    scale = abs_max / 7.0  # INT4 范围 [-8, 7],对称量化用 7
    W_q = torch.round(W / scale).clamp(-8, 7).to(torch.int8)
    return W_q, scale.to(torch.bfloat16)

def quantize_specialist(layer_idx, slot_j, perms):
    quantized = []
    for e in range(64):
        orig_idx = perms[e][slot_j]
        W_up = expert.up_proj.weight[orig_idx]      # (d_model,)
        W_gate = expert.gate_proj.weight[orig_idx]
        W_down = expert.down_proj.weight[:, orig_idx]
        
        W_up_q, s_up = quantize_int4_symmetric(W_up)
        # ...
        quantized.append({
            'W_up': (W_up_q, s_up),
            'W_gate': (W_gate_q, s_gate),
            'W_down': (W_down_q, s_down),
        })
    return quantized
```

**改进版(ablation)**:per-channel 而不是 per-tensor 量化(scale 按 d_model 维度分块计算)。**先做 per-tensor,简单优先**。

### 5.4 输出
对每层 × 每压缩率:
- `outputs/compressed_weights/layer{L:02d}_B{compression}/universal.pt`
  - dict: `{slot_j: {'W_up': tensor, 'W_gate': tensor, 'W_down': tensor}}`
- `outputs/compressed_weights/layer{L:02d}_B{compression}/group.pt`
  - dict: `{slot_j: {'W_up': (G, d_model), ..., 'assignment': (64,)}}`
- `outputs/compressed_weights/layer{L:02d}_B{compression}/specialist.pt`
  - dict: `{slot_j: {e: {'W_up': (int8, scale), ...}}}`

### 5.5 Sanity check
- 总存储量(累加所有 tier 的实际 byte 数)应当 ≈ target_compression × 原始大小,偏差 < 5%
- Universal 权重的范数应当在合理范围(不应该接近 0,也不应该爆炸)

---

## Phase 6: 前向重构(Forward Reconstruction)

### 6.1 数学公式

原始 MoE 层(对一个 token x,路由到 top-K 个 expert):
```
y = Σ_{e ∈ top-K} g_e · E_e(x)
E_e(x) = W_down^e @ (SiLU(W_gate^e @ x) ⊙ (W_up^e @ x))
       = Σ_{j=0..d_ff-1} W_down^e[:, j] × SiLU(W_gate^e[j,:] @ x) × W_up^e[j,:] @ x
       = Σ_j c_{e,j}(x)
```

其中 `c_{e,j}(x) = W_down^e[:, j] × SiLU(<W_gate^e[j,:], x>) × <W_up^e[j,:], x>` 是 expert e 第 j 个 neuron 对输出的贡献。

**压缩后的 forward**:把 j 按 tier 分类(用 `tier_map`):

```
y = Σ_{j ∈ Universal} K · c_U,j(x) · (mean gating factor)
   + Σ_{j ∈ Group}    Σ_{g ∈ active groups} count_g(top-K) · c_{g(j),j}(x) · (gating)
   + Σ_{j ∈ Specialist} Σ_{e ∈ top-K} g_e · c_{e,j}(x)
```

精确写法:
```python
y = 0
for j in universal_slots:
    c_U_j = W_down_U[:, j] * SiLU(<W_gate_U[j], x>) * <W_up_U[j], x>
    y += sum(g_e for e in top_K) * c_U_j  # 对所有 top-K expert 都相同

for j in group_slots:
    for g_id in unique_groups_in_top_K:
        experts_in_g = [e for e in top_K if group_assignment[e] == g_id]
        c_G_j = W_down_G[g_id, :, j] * SiLU(<W_gate_G[g_id, j], x>) * <W_up_G[g_id, j], x>
        y += sum(g_e for e in experts_in_g) * c_G_j

for j in specialist_slots:
    for e in top_K:
        W_up_j_e = dequantize(W_up_S[j][e])  # INT4 → BF16
        # ... 同上
        c_S_j_e = W_down_e[:, j] * SiLU(<W_gate_e[j], x>) * <W_up_e[j], x>
        y += g_e * c_S_j_e
```

### 6.2 实现策略

**关键:不要重写整个 MoE forward,monkey-patch 一个 expert 类**

```python
class CompressedExpertGroup(nn.Module):
    """替换原 MoE 层中的 self.experts 列表(整个 group)"""
    def __init__(self, universal_weights, group_weights, specialist_weights,
                 tier_map, group_assignment, num_experts=64):
        super().__init__()
        # 注册所有压缩后的权重为 Parameter / Buffer
        # universal_weights: dict of slot → {'W_up', 'W_gate', 'W_down'}
        # group_weights: dict of slot → {'W_up': (G, d_model), ..., 'assignment': (G,)}
        # specialist_weights: dict of slot → {e: ...}
        # tier_map: (d_ff,) int
        # group_assignment: (num_experts,)
    
    def forward(self, x, router_logits, top_k_indices, top_k_weights):
        """
        x: (B*T, d_model)
        top_k_indices: (B*T, top_K) — 每个 token 路由到哪些 expert
        top_k_weights: (B*T, top_K) — gating 权重
        """
        # 三段分别计算
        y_universal = self._forward_universal(x, top_k_weights)       # (B*T, d_model)
        y_group     = self._forward_group(x, top_k_indices, top_k_weights)
        y_specialist = self._forward_specialist(x, top_k_indices, top_k_weights)
        return y_universal + y_group + y_specialist
```

**Universal 加速点**:`_forward_universal` 中所有 universal slot 的 `W_up_U @ x` 和 `W_gate_U @ x` **不依赖 expert**,可以一次性算完。这是论文里"universal compute reuse"的核心红利。

**Group 加速点**:同 group 的 expert 共享 group slot 计算,所以最多算 G 份。

**Specialist 慢点**:必须 per-expert 算,但因为 specialist 占比应该 < 50%,总开销可接受。

### 6.3 Monkey patch

```python
def patch_model_with_compressed_moe(model, compressed_layers):
    """
    compressed_layers: dict of {layer_idx: CompressedExpertGroup}
    """
    for layer_idx, compressed_group in compressed_layers.items():
        moe_layer = model.layers[layer_idx].mlp
        # 替换 experts 模块
        moe_layer.experts = compressed_group
        # 重写 forward 以使用新的 expert group
        # (DeepSeek-V2-Lite 的 MoE forward 需要研究,可能要 monkey-patch moe_layer.forward 整个)
    return model
```

**注意**:DeepSeek-V2-Lite 的 MoE forward 在 transformers 4.45 中已有实现,但 router + experts 是分开调用的。需要研究 `DeepseekV2MoE.forward` 的具体写法,然后做对应替换。这一步是最容易卡住的工程点,**预留 1-1.5 天**。

### 6.4 PPL 验证

```python
def evaluate_ppl(model, dataset='wikitext-2-raw-v1', max_samples=500):
    from datasets import load_dataset
    data = load_dataset('wikitext', dataset, split='test')
    text = '\n\n'.join(data['text'][:max_samples])
    encodings = tokenizer(text, return_tensors='pt')
    
    max_length = 2048
    stride = 1024
    nll_sum = 0.0
    n_tokens = 0
    
    with torch.no_grad():
        for begin in range(0, encodings.input_ids.size(1), stride):
            end = min(begin + max_length, encodings.input_ids.size(1))
            input_ids = encodings.input_ids[:, begin:end].cuda()
            
            outputs = model(input_ids, labels=input_ids)
            nll_sum += outputs.loss.item() * (end - begin)
            n_tokens += (end - begin)
    
    return np.exp(nll_sum / n_tokens)
```

### 6.5 输出
- `outputs/ppl_results.json`:
```json
{
  "dense_baseline": 5.62,
  "uniform_int4": 6.21,
  "fans_moe": {
    "0.4": 5.85,
    "0.5": 5.74,
    "0.6": 5.68
  }
}
```

---

## Phase 7: 运行顺序与时间预算

```bash
# run_all.sh
python src/extract_activations.py --layers 1-26 --samples 256 --tokens 512  # ~4-6h
python src/align_neurons.py --layers 1-26                                    # ~6h
python src/compute_distance.py --layers 1-26                                 # ~3h
python src/tier_allocation.py --layers 1-26 --budgets 0.4,0.5,0.6           # ~10min
python src/compress_tiers.py --layers 1-26 --budgets 0.4,0.5,0.6            # ~30min
python src/forward_reconstruct.py --budgets 0.4,0.5,0.6                     # ~2h(含 PPL)
```

**总耗时 ~15-18 小时**,正好两天能跑完(允许失败重试)。

---

## 关键风险与回退方案

1. **Activation 存储爆磁盘**:缩减到 64 样本 × 256 tokens,或只对前 10 层做完整分析(把它当作 representative subset)。

2. **Hungarian 跑得太慢**:切换到 greedy 贪心。代码已经留了入口。

3. **DeepSeek-V2-Lite 的 MoE forward monkey-patch 卡住**:回退方案——不替换 forward,只把权重替换成"压缩-解压"后的等价 BF16 权重(失去 compute 红利但能拿到 PPL 数,作为 algorithmic-only 验证)。

4. **PPL 完全崩坏(>50)**:检查 alignment 是否正确(用 sanity check),检查 compression formula 是否对齐(权重维度对不对)。如果 alignment 是问题,先用 identity permutation 跑一遍 ablation,看到什么程度。

5. **Universal 占比异常低**(<5%):说明 tau1 太严或 D 矩阵分布异常,放宽 tau1,或检查 calibration data 是否多样性不足。

---

## 输出给论文的关键 figure/table

实验跑完后,Codex 需要生成:

1. **Figure 1**:跨层 tier 分布堆叠柱状图(浅层 vs 深层)
2. **Figure 2**:D_j 的直方图,叠加 tau1/tau2 阈值线
3. **Table 1**:PPL 主结果(dense / uniform INT4 / FANS-MoE @ 3 个压缩率)
4. **Table 2**:Ablation——三档 vs 二档,功能距离 vs 权重距离
5. **Figure 3**(可选):universal slot 的 compute reuse 收益(理论 FLOPs 节省曲线)

---

## 与 Codex 协作的注意事项

- 每个 phase 完成后 **必须运行 sanity check**,不通过不进入下一 phase
- 所有中间产物(activations, alignments, distances, tier_maps, compressed_weights)都要保存,方便从任意 phase 重启
- 用 `tqdm` 显示进度条
- 用 `wandb` 或简单 csv 记录每 phase 的关键指标
- 关键参数(`tau1`, `tau2`, `G`, `top_K_used_for_compression`)都从 `configs/deepseek_v2_lite.yaml` 读取,不要硬编码
