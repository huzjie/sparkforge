# FAQ

**Q: 是真实权重吗？** 不是。这是「机制参考实现」——用确定性的 mock 复现架构与调度逻辑，
不含 Muse Spark Open 权重（权重请到 Hugging Face 官方仓库下载）。

**Q: 为什么声明 816B 却不实例化 816B 的 tensor？** 纯 Python 实例化万亿级参数会卡死。
声明规模与真实 tensor 解耦：架构形状（256 专家 / top-8 / 1M 上下文 / 统一内存）是真实
的，规模用 `moe.total_params_b` 表达。

**Q: 能接真实后端吗？** `backends/openai.py` 通过 `SPARKFORGE_OPENAI_BASE` 环境变量
对接 OpenAI 兼容端点；vllm/transformers/llamacpp 留了适配器骨架，填上真实推理代码即可。

**Q: 训练信号为什么「单调趋 1」？** mock 的 `boost=(skill-0.5)*2` 在 skill=0.5 时恰好为
0（随机基线），skill=1.0 时恒命中 gold，所以路由 recall 从 baseline 干净爬到 1.0——这是
为了演示曲线可读，不是真实训练。
