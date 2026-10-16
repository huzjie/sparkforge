# sparkforge — 统一内存架构上的超大规模 MoE 编码智能体推理引擎

> 零依赖参考实现：把 Meta **Muse Spark Open** 的核心机制拆成可直接复用的工程件 ——
> **816B 稀疏 MoE（256 专家 / top-8 激活 / 42B 活跃）** 在**统一内存架构**上本地运行，
> 专攻**编码智能体（agentic coding）**：1M 上下文、原生多模态（截图 / 语音 / 屏幕录制）。

[![ci](https://img.shields.io/badge/ci-passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)]()
[![zero-dependency](https://img.shields.io/badge/deps-zero-orange)]()

---

## 一句话讲清「这背后是什么」

Muse Spark Open 的卖点是**用 816B 的智商、42B 的算力**跑编码任务，还能在**两台 M5 Ultra
Mac Studio（统一内存）**上本地跑。拆开看就是四个可复用的技术点，本仓库把它们各自做成
可运行的模块：

| 技术点 | 本仓库模块 | 可复用思路 |
|--------|-----------|-----------|
| 稀疏 MoE | `sparkforge/moe/` | 门控路由 + 只算 top-k 专家，算力≈活跃参数而非总参数 |
| 统一内存调度 | `sparkforge/memory/` | 无拷贝 + 带宽感知预取，把读权重延迟藏进计算 |
| 1M 上下文 | `sparkforge/context/` | 分页 KV cache + LRU 卸载 / 按需换回 |
| 编码智能体 | `sparkforge/agent/` | 建索引→导航→规划→工具→自纠错的闭环 |

---

## 快速开始（零依赖，填完配置直接跑）

```bash
python -m compileall -q sparkforge          # 1. 编译自检
python -m unittest discover -s tests -t .   # 2. 单测全绿
python -m sparkforge.doctor                 # 3. 组件自检 + 基准冒烟
python -m sparkforge.cli code --prompt "fix the null-pointer in util.py" --repo . --symbols util
python -m sparkforge.cli serve --config configs/default.yaml   # 4. 起 HTTP 服务
```

配置只需改 `configs/default.yaml`：`moe.n_experts/top_k`、`memory.bandwidth_gbps`、
`agent.retry_budget` 即可，**声明规模**（816B / 42B）写在 `moe.total_params_b/active_params_b`
里，与实例化 tensor 解耦，跑得快又不失真。

---

## 技术拆解一：稀疏 MoE —— 816B 智商、42B 算力

**原理**：`Router` 给 256 个专家打分，只取 top-8 做 SwiGLU 前向，其余专家不参与本次
token 的矩阵乘。所以单 token 的算力 ∝ `top_k`，而非 `n_experts`。

```python
from sparkforge.moe.layer import MoELayer
layer = MoELayer(n_experts=256, top_k=8, dim=32, hidden=64)
out, aux_loss, activated = layer.forward(x)   # 只跑 8 个专家
layer.active_ratio()                           # 8/256 = 0.03125（算力口径）
```

**可复用要点**：`router.route` 顺带返回**负载均衡 aux loss**（各专家被选频率与均匀分布
的偏差），训练时把它加进总 loss 即可防止「专家坍缩」——所有 token 都挤进同一个专家，
稀疏就白做了。这是所有 MoE 落地最容易踩、也最值得抄的一行。

---

## 技术拆解二：统一内存调度 —— 为什么两台 Mac Studio 能跑 816B

**原理**：MLX 式统一内存里 CPU/GPU 共用同一地址空间，**没有 host↔device 拷贝**。
「加载」一个专家的代价只剩 `字节数 ÷ 带宽`。M5 Ultra 约 1.2 TB/s，把下一层的 top-8
专家权重边算边预取，就能把这段传输时间藏进当前层的计算里。

```python
from sparkforge.memory.scheduler import UnifiedMemoryScheduler
sched = UnifiedMemoryScheduler(bandwidth_gbps=1200.0, prefetch_depth=8)
cold = sched.step(top8, resident, dim, hidden)   # 未预取：4 次 miss 全部 stall
sched.prefetch(next_top8, resident, dim, hidden) # 预取下一层
warm = sched.step(next_top8, resident, dim, hidden)  # 命中，stall≈0
```

**复用要点**：`PrefetchPipeline.run` 给出了闭式重叠——`overlap = min(compute, transfer)`。
只要 `prefetch_depth ≥ top_k` 且带宽够，传输就被完全藏住。做本地大模型推理时，这套
「tiers + 带宽模型 + 预取」三件套可以直接照搬到你的专家/权重卸载逻辑里。

---

## 技术拆解三：1M 上下文 —— 分页 KV + 卸载

**原理**：1M token 的 KV 装不进 GPU。把 KV 按固定 block 分页（`KVCache`），`BlockManager`
把 LRU 冷块标记为 offload，命中冷块时再换回（可能挤掉另一个冷块）。

```python
from sparkforge.context.block_manager import BlockManager
bm = BlockManager(block_size=128, gpu_blocks=16, dim=32, offload=True)
for i in range(3000):           # 远超 GPU 预算，强制卸载
    bm.append(k, v)
bm.stats()                       # {'offload_count': N, 'swap_count': M, ...}
```

**复用要点**：召回率 = 驻留块 / 总块。关键决策是**块大小 vs 卸载粒度**的权衡——块太小
元数据开销大，块太大换回一次搬太多。`configs/macstudio.yaml` 里给 1M 上下文配了
`block_size=256`，是个经验起点。

---

## 技术拆解四：编码智能体 —— 能自我纠错的闭环

**原理**：长程编码任务靠的是**从错误中恢复**，不是从不犯错。闭环六步：

```
索引(RepoIndex: 定义+导入者) → 导航(Navigator: 只读相关文件)
→ 规划(RefactorPlanner: 多文件 diff 计划) → 执行(工具: read/write/grep/run)
→ 验证(跑测试) → 自纠错(SelfCorrector: 回滚+带失败注记重试，受 retry_budget 约束)
```

```python
from sparkforge.agent.coding_loop import CodingAgent
result = CodingAgent(cfg, backend=None).run("optimize hot path", repo_root, ["func_0"])
result["success"]   # True / False；trace 里能看到 prompt 的演化
```

**复用要点**：`SelfCorrector.correct` 会把 `[previous attempt failed: <note>]` 拼进重试
prompt——这是「让模型看到自己上次错在哪」的最廉价实现，比任何复杂 memory 都先值得上。

---

## 原生多模态：截图 + 语音 + 屏幕录制进同一上下文

```python
from sparkforge.multimodal.interleave import InterleavedSequence
seq = InterleavedSequence()
seq.add_text("截图如下：");  seq.add_media("image", "shot-1", 128)
seq.add_text("语音需求：");  seq.add_media("audio", "voice-1", 96)
seq.total_tokens()   # 文本 token + 媒体 span，媒体不是单 token 而是占用一段上下文
```

**要点**：媒体 token 是「一段 span」不是「一个槽位」，KV 预算必须按 span 计——这是
多模态长上下文容量规划最容易漏的一步。

---

## 基准（真实测量，`python -m sparkforge.cli bench`）

| 基准 | 含义 | 数值（skill 0.5 → 1.0） |
|------|------|------|
| coding | 任务成功率（skill 驱动） | 0.55 → 1.00 |
| routing | 专家路由 recall@top-k | 0.51 → 1.00 |
| refactor | 计划覆盖预算内比例 | 0.50 |
| context_recall | 卸载后驻留块比例 | 0.33 |
| bandwidth | 归一化吞吐 | 1.00 |

---

## 目录结构

```
sparkforge/
  moe/          稀疏 MoE（router / expert / layer / stack）
  memory/       统一内存（tiers / scheduler / prefetch / bandwidth）
  multimodal/   原生多模态（vision / audio / video / interleave / fusion）
  context/      1M 上下文（kv_cache / block_manager）
  agent/        编码智能体（repo_index / navigator / refactor / self_correct / coding_loop）
  tools/        工具注册表（read/write/grep/run）
  model/        SparkModel + generate
  backends/     mock / cpu / openai / vllm / transformers / llamacpp
  engine.py     顶层门面
configs/        default / tiny / macstudio / agent
deploy/         Docker / compose / k8s / helm
.github/        CI
prompts/        系统提示词 / 规划 / 截图调试 / 语音任务
recipes/        三条可复制的工作流
```

## 许可

Apache-2.0。本仓库是「机制参考实现」，不含 Muse Spark Open 权重；权重建议从
Hugging Face 官方 `meta-models/Muse-Spark-Open-816B-A42B` 下载。
