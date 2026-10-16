# Design decisions

1. **零依赖** — 内置极简 YAML 解析器（`utils/yamlish.py`）、纯 Python tensor、stdlib HTTP，
   任何环境直接跑，不装 PyTorch/numpy。
2. **声明规模与实例化解耦** — `moe.total_params_b` 表达 816B，但 forward 用小维度，
   避免纯 Python 卡死（历史踩坑：大维度 MoE 实例化 2 分钟无输出）。
3. **共享 oracle** — `oracle.py` 是 mock 与 benchmark 的共同真值来源，skill=1.0 必满分，
   保证数字可信。
4. **确定性优先** — md5 种子驱动随机，测试与基准可复现。
5. **mock 后端可训练** — `train_step` 单调提升 skill，训练演示无需真实优化器。
