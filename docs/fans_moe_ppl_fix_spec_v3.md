# FANS-MoE PPL Harness 修复规范(v3)

> **唯一目标:把 dense baseline PPL 从当前的 ~132000 修到个位数(理想 6-10,可接受 <15)。**
> 在 dense baseline 修对之前,所有压缩 PPL 数字一律无效,不许进 paper。
>
> 背景:上一轮 real-text 实验里,dense baseline PPL = 132327,mean NLL = 11.79。
> DeepSeek-V2-Lite 词表 ~102400,ln(102400)≈11.54。**当前 NLL 比均匀随机猜词还差**,
> 说明 forward 输出的是垃圾 logits,或 label 完全没对齐。这不是压缩的问题——
> 没压缩的 dense baseline 也坏,所以是评估管道本身坏了。

---

## 0. 硬件假设

可用 **2 张 RTX 5090(共 64GB 显存)**。利用这一点的目的**不是提速,是为了能不 offload 地干净装下模型**——这很可能就是 harness 坏掉的根因。

---

## 1. 三个最可能的坏因(按概率排序)

1. **激进 disk/CPU offload 破坏 forward**(最可疑)
   上轮用了 `--max-gpu-memory 18GiB` + `--offload-dir`,把 31GB 的 BF16 模型硬塞进 18GB 并 offload 到 disk。这种激进 offload 容易导致 dtype/device 错位,logits 变垃圾。
2. **label 对齐错误**——NLL 比均匀随机还差,典型是 causal LM 的 label 没右移,或 loss 把 context 部分也算进去了。
3. **eval 文本 `local_docs` 喂入方式错误**——读成二进制 / 拼接错 / 非连贯文本。

修复策略:**逐个排除,每步带护栏,baseline 不到个位数就停下报告,不要继续往下跑压缩。**

---

## 2. 修复步骤(严格按顺序)

### Step 1:换干净的加载方式(消除 offload)

用两张卡做 model parallel,**彻底删除所有 disk offload 参数**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "30GiB", 1: "30GiB"},   # 两张 5090,各留 2GB 余量
    trust_remote_code=True,
    # 不要任何 offload_folder / offload_state_dict
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True
)
```

**护栏 1**:加载后打印 device map,确认没有任何模块落在 `disk` 或 `cpu`:
```python
from collections import Counter
devs = Counter(str(v) for v in model.hf_device_map.values())
print("DEVICE MAP:", devs)
assert not any("disk" in str(d) for d in model.hf_device_map.values()), \
    "STILL OFFLOADING TO DISK — abort"
```
若两张卡仍装不下(出现 cpu/disk),改用 FP8 加载(`torch_dtype` 换 `torch.float8_e4m3fn` 或用 bitsandbytes 8bit),但**优先 BF16 + 双卡**。

### Step 2:换标准 eval 数据(wikitext-2,弃用 local_docs)

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in test["text"] if t.strip())
enc = tokenizer(text, return_tensors="pt")
print("EVAL TOKENS:", enc.input_ids.shape[1])   # 应当 ~280000
```

**护栏 2**:确认 eval token 数 > 50000(local_docs 那 510 token 太少,统计噪声极大)。本轮至少评估 **20000 token**(够稳,又不至于太慢)。

镜像拉不动时 fallback:`wikitext-103-raw-v1` 的 test,或任意 >1MB 的连贯英文 txt。

### Step 3:正确的滑窗 PPL(带 label 对齐自检)

标准 strided perplexity,**关键是 target_ids 把 context 部分 mask 成 -100**:

```python
import torch, math

def compute_ppl(model, enc, max_length=2048, stride=1024, max_tokens=20000):
    input_ids_full = enc.input_ids
    seq_len = min(input_ids_full.size(1), max_tokens)
    nll_sum, n_tokens = 0.0, 0
    prev_end = 0
    first_device = next(iter(model.hf_device_map.values()))
    dev = f"cuda:{first_device}" if isinstance(first_device, int) else "cuda:0"

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end          # 这一窗里"新"的 token 数
        input_ids = input_ids_full[:, begin:end].to(dev)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100   # 只对新 token 算 loss,context 部分 mask 掉

        with torch.no_grad():
            out = model(input_ids, labels=target_ids)
            # out.loss 是这一窗 *有效 token* 的平均 NLL
            num_valid = (target_ids != -100).sum().item() - \
                        (target_ids[:, :-trg_len] == -100).sum().item() * 0  # = trg_len-ish
            # 更稳的算法:直接用 loss * 有效 token 数
            valid = (target_ids[..., 1:] != -100).sum().item()
            nll_sum += out.loss.item() * valid
            n_tokens += valid

        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(nll_sum / n_tokens)
    return ppl, nll_sum / n_tokens, n_tokens
```

**护栏 3(label 对齐自检,关键)**:用一个已知健康的小模型(如 `gpt2`)先跑一遍这个 `compute_ppl`,wikitext-2 上 gpt2 的 PPL 应当 ~29-30。**如果连 gpt2 都跑出 >100,说明是 PPL 函数本身的 bug,而不是 DeepSeek 加载问题**——先修函数。这一步能在 1 分钟内把"加载 bug"和"评估函数 bug"区分开,务必做。

```python
# 自检:gpt2 在 wikitext-2 应当 PPL ~29
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
enc_gpt2 = gpt2_tok("\n\n".join(t for t in test["text"] if t.strip()), return_tensors="pt")
ppl_check, _, _ = compute_ppl(gpt2, enc_gpt2, max_length=1024, stride=512, max_tokens=20000)
print(f"GPT2 SANITY PPL = {ppl_check:.2f}  (expect ~25-35)")
assert ppl_check < 50, "PPL FUNCTION IS BROKEN — fix the function before touching DeepSeek"
```

### Step 4:测 DeepSeek dense baseline,设硬护栏

```python
ppl, nll, ntok = compute_ppl(model, enc, max_length=2048, stride=1024, max_tokens=20000)
print(f"DENSE BASELINE PPL = {ppl:.3f}  NLL = {nll:.4f}  tokens = {ntok}")

# 硬护栏:baseline 不到 15 就停,不要继续跑压缩
assert ppl < 15, f"DENSE BASELINE STILL BROKEN (ppl={ppl:.1f}) — STOP, report to user"
```

**baseline < 15 之前,不要进入 Step 5。** 如果两张卡 + 无 offload + wikitext + gpt2 自检通过后 baseline 仍 >15,把以下信息打包报告:device_map、gpt2 自检 PPL、eval token 数、DeepSeek 前 5 个 token 的 top-1 预测(看是不是输出乱码)。

---

## 3. baseline 修好后才做:压缩 PPL + INT4 baseline

### Step 5:三档 PPL(沿用 v2 的 fallback_bf16 重构路径)
- 用已生成的 tier maps(`outputs_realtext_512/tier_maps`)
- budgets:0.08 / 0.12 / 0.18
- 重构方式:压缩→解压回 BF16,塞回模型,跑同一个 `compute_ppl`
- **护栏 4**:每档 PPL 必须 (a) <20,(b) 随压缩变狠单调上升。不满足就是重构 bug。

### Step 6:补 uniform INT4 baseline(这次不许 skip)
- 把所有 routed expert 的所有 neuron 一律 INT4 量化(不分档),测 PPL
- 这是审稿人判断"三档 vs 无脑全 INT4"的关键对照
- 填进结果:`uniform_int4_allexpert`

### 最终输出 `ppl_results_v3.json`
```json
{
  "gpt2_sanity_ppl": <~29>,
  "eval_dataset": "wikitext-2-raw-v1",
  "eval_tokens": <~20000>,
  "dense_baseline_ppl": <个位数>,
  "uniform_int4_allexpert_ppl": <数>,
  "fans_moe": {"0.08": <数>, "0.12": <数>, "0.18": <数>}
}
```

---

## 4. 顺手补两件 paper 要用的小东西

### 4.1 四层 D 直方图(判断分层故事怎么讲)
对 layer 1/5/13/26,把 `distances/layerXX_D.npy` 画成直方图(20-30 bins),叠加 tau1/tau2 切点线,存 `outputs/figs/D_hist_layerXX.png`。**我要用这个判断 D 分布是双峰还是单峰长尾。**

### 4.2 shared expert 对齐分析(paper intro 钩子)
- 取 routed expert 中划归 universal 的 neuron 的功能签名
- 算它们与 2 个 shared expert 的 neuron 签名的平均 cos 相似度
- 对照:specialist neuron 与 shared expert 的相似度(应当更低)
- 输出 `outputs/shared_alignment.json`:`{universal_vs_shared: <数>, specialist_vs_shared: <数>}`
- **预期**:universal_vs_shared 显著 > specialist_vs_shared,支撑"universal neuron 在功能上接近 DeepSeek 显式设计的 shared expert"这一论点

---

## 5. 执行命令骨架

```bash
cd ~/workspace/fans_moe
. ../venv-dsv2/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2,3          # 两张卡

# Step 1-4: 修 baseline(带 gpt2 自检 + 双卡无 offload + wikitext)
python -u src/fix_ppl_harness.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --eval-dataset wikitext-2-raw-v1 \
    --eval-tokens 20000 \
    --max-memory-per-gpu 30GiB \
    --gpt2-sanity \
    --baseline-only

# baseline < 15 通过后,再跑压缩 + INT4
python -u src/fix_ppl_harness.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --eval-dataset wikitext-2-raw-v1 \
    --eval-tokens 20000 \
    --max-memory-per-gpu 30GiB \
    --weights-dir outputs_realtext_512 \
    --budgets 0.08,0.12,0.18 \
    --with-uniform-int4 \
    --output ppl_results_v3.json
```

---

## 6. 不许再犯的错

1. **不许用 disk offload** —— 两张卡装得下,装不下就 FP8,绝不 offload 到 disk
2. **不许用 local_docs 当 eval** —— 用 wikitext-2-raw-v1
3. **不许 eval token <50000 的统计**(本轮取 20000 是最低线,510 太荒谬)
4. **不许跳过 gpt2 自检** —— 它 1 分钟就能区分"加载 bug"还是"PPL 函数 bug"
5. **baseline 不到 15 不许跑压缩** —— 带病往下全是浪费
6. **不许 skip uniform INT4 baseline** —— 这是核心对照

---

## 7. 跑完回报清单(贴给我判断能否 freeze 算法层)

1. `gpt2_sanity_ppl`(应 ~29)—— 证明 PPL 函数对
2. `dense_baseline_ppl`(应个位数)—— 证明加载/评估对
3. `uniform_int4_allexpert_ppl` 和三档 `fans_moe` PPL —— 看三档是否优于无脑 INT4
4. 四层 `D_hist_layerXX.png` —— 判断分层故事
5. `shared_alignment.json` —— 判断 intro 钩子是否成立
