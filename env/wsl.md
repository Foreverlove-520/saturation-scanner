
## 已知依赖冲突（2026-10-01 实测）
SGLang v0.5.12 + transformers 5.6.0 + kernels 0.17.1 冲突：
import sglang → transformers.integrations.hub_kernels →
kernels/layer/layer.py:92 LayerRepository.__init__ 抛
"ValueError: Either a revision or a version must be specified."
解法：uv pip uninstall kernels（USE_HUB_KERNELS=0 无效，开关拦不到 import 路径）
影响：Hub kernel 关闭，退回参考实现。SGLang 核心路径不经 transformers，
     视为实验条件之一，已在 README 声明。

## "能启动" ≠ "能用"（2026-10-03）
现象：server 打印 ready to roll、/health 返回 200、warmup POST /generate 200，
      但第一个真实请求返回 HTTP 000。
原因：sampling_backend 默认 flashinfer，真实请求时才触发 JIT 编译，
      撞 CUDA 工具链问题 → 进程 SIGQUIT 崩溃。
      （warmup 用 greedy 采样，不触发该 JIT，所以"看起来正常"）
解法：--sampling-backend pytorch + 补 -lcuda 软链
教训：验收必须发真实请求，不能只看启动日志和 /health。
