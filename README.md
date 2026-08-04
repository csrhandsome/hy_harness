# Hy-Harness

Hy-Harness 是基于 Hy-VLA 的 LIBERO Harness 运行环境。本 README 只说明运行所需的环境、模型权重、Benchmark 资源和 vLLM Serving 配置。

## 1. 环境要求

- Linux
- Python 3.10–3.12
- NVIDIA GPU、CUDA 驱动和可用的 MuJoCo/EGL 图形环境
- [uv](https://docs.astral.sh/uv/)
- Git、`unzip`

安装 `uv`（如果尚未安装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. 获取代码

```bash
git clone https://github.com/csrhandsome/hy_harness.git
cd hy_harness
```

## 3. 下载模型权重

Harness 使用的是面向 LIBERO 的 Hy-VLA checkpoint。请准备一个本地 checkpoint 目录，并确保至少包含：

```text
<HY_VLA_CHECKPOINT>/
├── config.json
├── *.safetensors 或 *.bin
└── norm_stats.pkl
```

其中 `norm_stats.pkl` 需要包含以下字段：`qpos_mean`、`qpos_std`、`action_mean` 和 `action_std`。如果归一化文件不在 checkpoint 目录中，可以通过 `NORM_PATH` 单独指定。

```bash
export CKPT_PATH=/absolute/path/to/hy-vla-libero
export HY_VLA_CHECKPOINT="$CKPT_PATH"
# 可选：export NORM_PATH=/absolute/path/to/norm_stats.pkl
```

注意：`libero_eval/download_libero_pro_ckpts.sh` 下载的是旧的 VLA-Adapter LIBERO-Pro checkpoint，不是 Hy-VLA Harness 所需的权重，不要将其作为 `CKPT_PATH` 使用。

## 4. 下载 Benchmark 资源

Benchmark 代码和大文件不提交到本仓库，需要放到以下固定目录：

运行 LIBERO 评测前，请在 `third_party/` 目录下分别 `git clone` 以下三个 LIBERO 环境仓库：

```text
third_party/
├── LIBERO/
├── LIBERO-plus/
└── LIBERO-PRO/
```

下载三个 Benchmark 的代码：

```bash
mkdir -p third_party

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  third_party/LIBERO
git clone https://github.com/sylvestf/LIBERO-plus.git \
  third_party/LIBERO-plus
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git \
  third_party/LIBERO-PRO
```

下载 LIBERO-plus 资源：

```bash
bash libero_eval/download_libero_plus.sh
```

下载 LIBERO-Pro 的 BDDL 和初始状态文件：

```bash
bash libero_eval/download_libero_pro.sh
```

下载 Harness 使用的 RPent memory 资源：

```bash
bash libero_eval/download_rpent_memory.sh
```

资源下载完成后的关键目录如下：

```text
third_party/LIBERO/libero/libero/
third_party/LIBERO-plus/libero/libero/assets/
third_party/LIBERO-PRO/libero/libero/{bddl_files,init_files}/
sde_harness/resources/libero/
```

默认下载使用 `https://hf-mirror.com`。网络不通时可以切换到 Hugging Face 官方站点：

```bash
HF_ENDPOINT=https://huggingface.co bash libero_eval/download_libero_plus.sh
HF_ENDPOINT=https://huggingface.co bash libero_eval/download_libero_pro.sh
HF_ENDPOINT=https://huggingface.co bash libero_eval/download_rpent_memory.sh
```

## 5. 配置根目录 Harness 环境

根目录使用 `uv` 管理环境。Harness 依赖定义在根目录 `pyproject.toml` 的 `harness` extra 中：

```bash
uv sync --extra harness
```

如果只需要基础 Hy-VLA 环境，可以使用：

```bash
uv sync
```

Harness 运行时会使用根目录环境；不需要在 `sde_harness/` 下单独创建虚拟环境。

Harness Planner 默认使用 CodeBuddy Agent SDK，并将请求导向本地 vLLM 服务。复制本地配置文件，确保本地 vLLM 已在 `http://127.0.0.1:8080/v1` 启动：

```bash
cp sde_harness/.env.local.example sde_harness/.env.local

# 默认通过 CodeBuddy SDK 访问本地 vLLM
export CODEBUDDY_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export CODEBUDDY_API_KEY=EMPTY
export CODEBUDDY_MODEL=hy_a3b
```

`.env.local` 只保存本机配置和密钥，不要提交到 Git。

### 5.1 独立仿真环境与动作模型服务

LIBERO 不再安装到根环境。创建独立仿真环境：

```bash
uv sync --project libero_eval
```

直接评测前，在根环境启动动作 Policy Server：

```bash
uv run --extra harness vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

然后在另一个终端运行：

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/run_libero_eval.sh
```

RoboTwin 也通过各自的轻量 client 请求根环境 Policy Server。完整环境边界、RoboTwin 命令和并发限制见 `docs/environment-layout.md`。

## 6. 配置 Serving/vLLM 环境

`serving` 使用独立的虚拟环境，因为它的 PyTorch 和 vLLM 版本与根目录环境不同。

```bash
cd serving
uv sync
cd ..
```

检查 Serving 配置和 Python/Shell 语法：

```bash
bash serving/scripts/verify_flow.sh
```

当前默认的 vLLM 模型是 `hy-embodied-vlm-1.0`，对应 Hugging Face 仓库 `tencent/Hy-Embodied-VLM-1.0`。下载权重到 `serving/cache/`：

```bash
bash serving/scripts/download_weights.sh --model hy-embodied-vlm-1.0
```

启动 OpenAI-compatible vLLM 服务：

```bash
bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0
```

默认配置为 Tensor Parallel 4 卡。如果使用单卡或需要覆盖显存配置：

```bash
TP=1 GPU_MEM_UTIL=0.85 \
  bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0
```

也可以直接指定本地权重目录：

```bash
MODEL_PATH=/absolute/path/to/Hy-Embodied-VLM-1.0 \
  bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0
```

服务启动后，可用以下命令检查接口：

```bash
bash serving/scripts/client.sh \
  --model hy-embodied-vlm-1.0 \
  --prompt "How do you open a fridge?"
```

CodeBuddy Agent SDK 默认通过本地 vLLM 调用模型，在 `sde_harness/.env.local` 中配置：

```bash
CODEBUDDY_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
CODEBUDDY_API_KEY=EMPTY
CODEBUDDY_MODEL=hy_a3b
```

`hy_a3b` 是 `serving/models/hy-embodied-vlm-1.0/model.yaml` 中配置的 served name。

## 7. 配置检查清单

开始运行前确认：

```text
[ ] CKPT_PATH 指向 LIBERO-finetuned Hy-VLA checkpoint
[ ] checkpoint 下存在 norm_stats.pkl，或已设置 NORM_PATH
[ ] 本地 vLLM 服务已启动，CodeBuddy SDK 默认连接 `http://127.0.0.1:8080/v1`，模型为 `hy_a3b`
[ ] third_party/LIBERO 已存在
[ ] third_party/LIBERO-plus 已存在且已下载 assets
[ ] third_party/LIBERO-PRO 已存在且已下载 BDDL/init 文件
[ ] sde_harness/resources/libero 已存在（Harness memory）
[ ] 根目录已执行 uv sync --extra harness
[ ] libero_eval 已执行 uv sync --project libero_eval
[ ] serving/ 已执行 uv sync
[ ] vLLM 权重已下载并且服务端口可访问
```

### 目录说明

```text
hy_harness/
├── .venv/                         # 根目录 Harness 环境，由 uv 管理
├── third_party/                   # Benchmark 代码和资源（Git ignored）
├── libero_eval/.venv/              # 独立 LIBERO 仿真环境
├── sde_harness/resources/         # Harness memory 资源（Git ignored）
├── serving/cache/                 # vLLM 权重缓存（Git ignored）
└── serving/.venv/                 # 独立 vLLM 环境，由 uv 管理
```

## 8. RoboTwin Harness（实验性）

现在可以在 RoboTwin 官方评测循环中启用一个嵌入式 Harness Toolkit。默认仍然使用原来的 Hy-VLA 直接策略；只有显式开启后，Planner 才会通过 `robotwin_observe`、`robotwin_vla_step` 和 `robotwin_execute_ee` 操作当前的 `TASK_ENV`。

先按照 RoboTwin 官方方式准备环境和任务资源，并将本目录链接为 RoboTwin 的 policy 目录：

```bash
ln -s /absolute/path/to/Hy-Embodied-0.5-VLA robotwin/policy/hy_vla
cd robotwin
```

在 `policy/hy_vla/deploy_policy.yml` 中打开：

```yaml
harness:
  enabled: true
  planner: codebuddy
  model: hy_a3b
  max_turns: 100
```

也可以临时启用：

```bash
ROBOTWIN_HARNESS=1 bash policy/hy_vla/eval.sh \
  <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

Planner 使用的凭据仍从 `sde_harness/.env.local` 或对应环境变量读取。每个 episode 的 Toolkit 动作记录和 Planner 结果写入 `logs/robotwin_harness/`。该接入目前是实验性的：它复用 RoboTwin 的官方 `TASK_ENV.get_obs()`、`get_instruction()` 和 `take_action(..., action_type="ee")` 接口，在第一次 `eval` 回调中运行完整 Planner；尚未提供独立的 `rpent.cli --env robotwin` 启动器，也没有替代 RoboTwin 自己的 episode/reset 管理。
