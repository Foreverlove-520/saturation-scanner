
## 已知依赖冲突（2026-10-01 实测）
SGLang v0.5.12 + transformers 5.6.0 + kernels 0.17.1 冲突：
import sglang → transformers.integrations.hub_kernels →
kernels/layer/layer.py:92 LayerRepository.__init__ 抛
"ValueError: Either a revision or a version must be specified."
解法：uv pip uninstall kernels（USE_HUB_KERNELS=0 无效，开关拦不到 import 路径）
影响：Hub kernel 关闭，退回参考实现。SGLang 核心路径不经 transformers，
     视为实验条件之一，已在 README 声明。
