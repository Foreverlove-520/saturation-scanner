# saturation-scanner

面向**消费级 GPU 显存受限场景**的推理引擎选型与调度策略研究。
在单张 RTX 4060 Laptop（8GB，可用约 6.5GB）上，实测多实例共存时的显存-性能权衡，
并探索任务感知调度能否缓解按显存静态切分导致的资源浪费。

---

## 一、硬件平台

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 **Laptop**（非台式版本） |
| 显存 | 标称 8 GB，**可用约 6.5 GB**（Windows 侧常驻约 1686 MiB） |
| 计算架构 | sm_89（compute capability 8.9） |
| 驱动版本 | 610.57.01 |
| 宿主系统 | Windows + WSL2 / Ubuntu 24.04 |

**三条已实测的硬件约束（直接影响实验设计）**：

1. **可用显存只有 6.5 GB 而非 8 GB** —— 双实例预算必须按 6.5 GB 重算
2. **`power.limit` 读取为 N/A** —— 锁功耗校验改用 CV ≤ 5% 判据
3. **GPU 时间戳错误**（`nvidia-smi` 显示 2026/1/1）—— 日志对齐必须用 WSL 侧 `date`

---

## 二、版本矩阵

| 组件 | 版本 | 备注 |
|---|---|---|
| SGLang | v0.5.12 (commit 127b9e3283) | 源码编译安装 |
| Python | 3.12 | venv ~/venv/sglang |
| PyTorch | 2.11.0+cu130 | 由 SGLang 依赖约束决定 |
| CUDA runtime | 13.0 | torch wheel 自带，未装完整 CUDA Toolkit |
| nvcc / ptxas / cicc | 13.0.88 | 三者必须同版本，否则 PTX 版本不匹配 |
| transformers | 5.6.0 | 由 pyproject 锁定 |
| flashinfer | 0.6.11 | 不可用，见下方已知冲突 |
| attention backend | **triton** | 全线统一使用 |
| sampling backend | **pytorch** | 必须显式指定 |
| Rust | 1.99.0 | 编译 SGLang router 需要 |
| 测试模型 | Qwen2.5-0.5B-Instruct | 988 MB，bf16 |

#### 已知依赖冲突（已解决，但必须声明为实验条件）

**1. transformers 5.6.0 与 kernels 冲突**

`import sglang` 时抛 ValueError。解法：`uv pip uninstall kernels`。
影响：Hub kernel 关闭，退回参考实现；SGLang 核心路径不经 transformers。

**2. flashinfer 无法构建**

需要 CUDA 13.1 以上才能编译，但 torch 被锁在 cu130，两者不可兼得。
解法：`--attention-backend triton` 且 `--sampling-backend pytorch`。
影响：attention 与 sampling 算子实现变更，所有实验组必须保持一致。

**3. "能启动" 不等于 "能用"**

曾出现 `ready to roll`、health 返回 200、warmup 返回 200，
但第一个真实请求返回 HTTP 000（进程崩溃）。
原因：sampling 默认走 flashinfer，真实请求才触发 JIT 编译。
教训：验收必须发真实请求，不能只看启动日志。

---

## 三、复现步骤

### 0. 环境变量（每次开新窗口都要）

    cd ~
    source ~/venv/sglang/bin/activate
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HUB_DISABLE_XET=1

### 1. 起 server

    nohup python -m sglang.launch_server \
      --model-path ~/models/qwen2.5-0.5b-instruct \
      --host 127.0.0.1 --port 30000 \
      --mem-fraction-static 0.35 \
      --served-model-name qwen05b \
      --attention-backend triton \
      --sampling-backend pytorch \
      > ~/sglang-30000.log 2>&1 &

    tail -f ~/sglang-30000.log   # 等到出现 ready to roll

两个 backend 参数都不能省略，原因见版本矩阵。
停止服务：pkill -f sglang.launch_server

### 2. 发请求

健康检查：

    curl http://127.0.0.1:30000/health

非流式：

    curl http://127.0.0.1:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen05b","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

流式（SSE）：

    curl http://127.0.0.1:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"qwen05b","messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}'

### 3. 起 router

待补（Go/No-Go 通过后填充）

---

## 四、验收结果（2026-10-03）

| 判据 | 结果 |
|---|---|
| 健康检查 | OK |
| 非流式 | 完整 JSON，返回 1+1=2 |
| 流式 SSE | 14 帧逐 token 输出 |
| 8 并发 | 8 个响应内容互不相同，无串行化、无串结果 |

### 首次启动实测数据

- KV cache：123392 tokens，K 与 V 各 0.71 GB
- 推算：12.07 KB/token，与手算 12.0 KB/token 吻合，验证显存占用可预测
- 模型权重：0.98 GB，分配后可用显存 4.15 GB

详细环境踩坑记录见 env/wsl.md
