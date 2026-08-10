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

注意：`bash libero_eval/download.sh pro-ckpts` 下载的是旧的 VLA-Adapter LIBERO-Pro checkpoint，不是 Hy-VLA Harness 所需的权重，不要将其作为 `CKPT_PATH` 使用。

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

统一下载 LIBERO-plus 和 LIBERO-Pro 评测资源：

```bash
bash libero_eval/download.sh
```

下载 Harness 使用的 RPent memory 资源：

```bash
bash sde_harness/scripts/download_rpent_memory.sh
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
HF_ENDPOINT=https://huggingface.co bash libero_eval/download.sh
HF_ENDPOINT=https://huggingface.co bash sde_harness/scripts/download_rpent_memory.sh
```

## 5. 配置根目录 Hy-VLA 环境

根目录环境只承担 Hy-VLA 的训练、本地推理和动作 Policy Server：

```bash
uv sync
```

Harness 的 Planner 依赖不再安装到根目录；它们由各自的仿真环境安装。`sde_harness/` 是共享源码包，不单独拥有虚拟环境。

Harness Planner 默认使用 CodeBuddy Agent SDK，并将请求导向本地 vLLM 服务。复制本地配置文件，确保本地 vLLM 已在 `http://127.0.0.1:8080/v1` 启动：

```bash
cp sde_harness/.env.local.example sde_harness/.env.local

# 默认通过 CodeBuddy SDK 访问本地 vLLM
export CODEBUDDY_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export CODEBUDDY_API_KEY=EMPTY
export CODEBUDDY_MODEL=hy_a3b
```

`.env.local` 只保存本机配置和密钥，不要提交到 Git。

### 5.1 LIBERO 仿真环境与 Harness

LIBERO 不再安装到根环境。创建独立仿真环境：

```bash
uv sync --project libero_eval
```

直接评测前，在根环境启动动作 Policy Server：

```bash
uv run vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

然后在另一个终端运行：

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/eval.sh
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
[ ] 根目录已执行 uv sync（训练、推理与 Policy Server）
[ ] libero_eval 已执行 uv sync --project libero_eval
[ ] serving/ 已执行 uv sync
[ ] vLLM 权重已下载并且服务端口可访问
```

### 目录说明

```text
hy_harness/
├── .venv/                         # 根目录 Hy-VLA 训练、推理与 Policy Server
├── third_party/                   # Benchmark 代码和资源（Git ignored）
├── libero_eval/.venv/              # 独立 LIBERO 仿真环境
├── sde_harness/resources/         # Harness memory 资源（Git ignored）
├── serving/cache/                 # vLLM 权重缓存（Git ignored）
└── serving/.venv/                 # 独立 vLLM 环境，由 uv 管理
```

## 8. RoboTwin Harness（实验性）

现在可以在 RoboTwin 官方评测循环中启用一个嵌入式 Harness Toolkit。默认仍然使用原来的 Hy-VLA 直接策略；只有显式开启后，Planner 才会通过 `robotwin_observe`、`robotwin_vla_step` 和 `robotwin_execute_ee` 操作当前的 `TASK_ENV`。

先按照 RoboTwin 官方方式准备相邻的 `../RoboTwin` checkout 和任务资源，并安装模拟器侧环境：

```bash
uv sync --project robotwin_eval
```

启动由三个独立入口组成；请分别在三个终端运行。`run_eval.sh` 会自动把仓库中的 `robotwin_eval/` 链接到 `RoboTwin/policy/hy_vla`，无需手工执行 `ln -s`。

```bash
# 终端 1：Harness planner VLM（含 tool-choice proxy）。
# 三个变量都不能省：SERVING_PYTHON 避开 cephfs 上的 venv（否则卡死）、
# GPU_MEM_UTIL=0.55 给 policy 腾显存、CUDA_VISIBLE_DEVICES 圈定 0-3 卡。
cd /apdcephfs_nj7/share_305204761/threecheng/hy_harness && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_MEM_UTIL=0.55 \
  SERVING_PYTHON=/tmp/hy-serving-venv/bin/python \
  bash serving/scripts/serve_harness.sh

# 终端 2：Hy-VLA action policy server 集群（16 路）。
# 启动时会先打印 gpus: 4,5,6,7,... (16 servers)，可据此确认变量生效。
cd /apdcephfs_nj7/share_305204761/threecheng/hy_harness && \
  GPUS="4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7" BASE_PORT=8001 \
  bash scripts/serve_policy_servers_robotwin.sh

# 终端 3a：先小规模验证（3 任务 × 5 集），确认 finish 时机与成功率后再上全量。
cd /apdcephfs_nj7/share_305204761/threecheng/hy_harness && \
  GPUS="4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7" BASE_PORT=8001 \
  TASKS="beat_block_hammer adjust_bottle blocks_ranking_rgb" TEST_NUM=5 \
  bash robotwin_eval/run_eval.sh

# 终端 3b：确认没问题后再全量（50 任务 × 20 集 = 1000 episode）。
cd /apdcephfs_nj7/share_305204761/threecheng/hy_harness && \
  GPUS="4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7" BASE_PORT=8001 TEST_NUM=20 \
  bash robotwin_eval/run_eval.sh
```

**注意事项**

- **终端 3 的 `GPUS` 必须和终端 2 逐字相同**——脚本按数组下标做 GPU→端口映射，不一致就会连错 server。
- 启动终端 3 后看第 2 行，应该是 `gpus: 4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7 (16 workers)`；如果显示 `0,1,2,3,4,5,6,7 (8 workers)`，说明 `GPUS` 没传进去（换行问题）。
- 想先空跑校验映射、不占 GPU：

  ```bash
  cd /apdcephfs_nj7/share_305204761/threecheng/hy_harness && \
    DRY_RUN=1 GPUS="4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7" BASE_PORT=8001 \
    bash robotwin_eval/run_eval.sh
  ```

- 三条命令都是前台阻塞的（前台进程会随终端结束）。要放后台：

  ```bash
  setsid nohup <命令> > logs/serving/xxx.log 2>&1 < /dev/null &
  ```

不设置 `TP` 或 `GPU_MEM_UTIL` 时，Planner VLM 使用 `serving/models/hy-embodied-vlm-1.0/model.yaml` 中的默认值（TP=4、显存利用率=0.85）；上例的 `GPU_MEM_UTIL=0.55` 是为同机 policy server 预留显存的显式覆盖。Planner 凭据仍从 `sde_harness/.env.local` 或对应环境变量读取。每个 episode 的 Toolkit 动作记录和 Planner 结果写入 `logs/robotwin_harness/`。该接入目前是实验性的：它复用 RoboTwin 的官方 `TASK_ENV.get_obs()`、`get_instruction()` 和 `take_action(..., action_type="ee")` 接口，在第一次 `eval` 回调中运行完整 Planner；尚未提供独立的 `rpent.cli --env robotwin` 启动器，也没有替代 RoboTwin 自己的 episode/reset 管理。目前只端到端验证过 1 个 episode（finish 出现在仅 1 次 `vla_step` 之后，基准侧判定 Success），所以先跑小规模看 finish 时机分布，比直接上 1000 集稳妥。