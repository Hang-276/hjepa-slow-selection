# H-JEPA：图像表征中的可靠慢因素选取

这个文件夹可以单独复制到服务器。正式实验需要 Linux、NVIDIA GPU、Python 3.11，以及可访问 GitHub、Hugging Face 和 PyTorch 下载站的网络。建议使用一张 24 GB 显存的 GPU、至少 16 GB 内存，并预留 30 GB 磁盘；这是资源建议，完整运行时间和峰值用量尚未在服务器实测。

**本实验读取官方已训练的 H-JEPA，冻结编码器，训练物理因素读出并评价两层表征。没有从头训练整个 H-JEPA。** 它检验低维研究的判断在视觉层次表征中是否仍成立，不复现原论文的完整预训练和规划实验。

## 1. 安装与运行

以下命令都在服务器上执行。先克隆仓库；这是私有仓库，需要服务器已经配置有访问权限的 GitHub 凭据。HTTPS 提示密码时使用 GitHub personal access token，不能使用账号密码；已配置 SSH 的用户也可使用 SSH 克隆地址。

```bash
git clone https://github.com/Hang-276/hjepa-slow-selection.git
cd hjepa-slow-selection
python3.11 --version
nvidia-smi
bash setup.sh
```

安装脚本创建本目录内的 `.venv`，下载固定版本的官方代码和三个官方模型。不会下载官方训练数据。安装前需确保 `python3.11 --version` 和 `nvidia-smi` 能正常运行，且 Python 包含 `venv` 支持。如果服务器有 Conda 但没有 Python 3.11，可先执行 `conda create -n hjepa-server python=3.11 -y`、`conda activate hjepa-server`，再执行 `bash setup.sh`。PyTorch 使用 CUDA 12.8 版本，需要服务器 NVIDIA 驱动支持；不要求单独安装 CUDA 编译工具链。

首次安装需下载依赖和三个权重，耗时取决于网络。若网络中断，可重新执行 `bash setup.sh`，已完成的下载会使用缓存。本机不需要下载这些文件。

先运行一次工程检查：

```bash
.venv/bin/python -m unittest discover -v -p 'test_*.py'
.venv/bin/python run.py --name smoke_01 --smoke --device cuda
```

看到 `results/smoke_01/status.json` 中的 `SMOKE_COMPLETE`，表示数据生成、官方模型读取、读出拟合和结果保存这条流程走通。小样本测试不用于判断方法有效，也不要求风险门槛通过。

正式运行：

```bash
mkdir -p results
mkdir results/fourroom_01 && nohup .venv/bin/python run.py --name fourroom_01 --device cuda > results/fourroom_01/launch.log 2>&1 &
```

这条命令退出 SSH 后仍会继续运行。查看进度：

```bash
tail -f results/fourroom_01/run.log
```

生成图像阶段的详细进度在 `results/fourroom_01/collection.log`。如果启动后没有 `run.log` 或 `status.json`，先查看 `results/fourroom_01/launch.log`，里面会保留依赖或 CUDA 检查等入口错误。按 Ctrl+C 退出 `tail` 只会关闭进度查看，不会终止正式实验。结束后查看：

```bash
cat results/fourroom_01/status.json
cat results/fourroom_01/RESULTS.md
```

`COMPLETE` 表示整套实验执行结束，**不表示科学假设通过**。具体结论看 `RESULTS.md` 和 `summary.csv`。若为 `FAILED`，保留整个结果目录，错误原因在 `run.log`；本脚本不自动重启或覆盖失败结果。

如果显存不足，可以在正式运行前把 `config.json` 中的 `encode_batch_size` 从 64 改为 32 或 16，再用新名称运行。它只改变图像编码批量。不要根据 TEST 结果修改读出、阈值、因素或统计门槛，再把修改后的运行当作原来的独立验证。

## 2. 结果全部从一个目录取走

本次实验的所有分析材料都在 `results/fourroom_01/`：

```text
results/fourroom_01/
  RESULTS.md                 人能直接阅读的结果表与状态解释
  summary.csv / summary.json 每个模型、层和阈值规则的完整统计
  status.json / run.log       完成状态、进度及错误记录
  launch.log                 后台运行的启动输出，包括入口错误
  config.json / scripts/     本次设置及代码副本
  provenance.json            官方版本、模型文件哈希、设备信息
  environment.txt            Python 依赖版本
  data.json / splits.json    数据哈希、各用途的 episode 与帧索引
  collection*                生成数据的命令、配置及日志
  thresholds.json            TEST 预测前固定的全部阈值
  forecast_receipt.json      读取 TEST 真值前保存的预测文件哈希
  reference_truth.npz        共同物理参考轨迹及方差
  test_truth.npz             TEST 物理轨迹
  seed42_level1/              其他模型与层采用相同目录结构
    probe.pt / probe.json    读出权重、标准化参数、拟合记录
    audit.json / audit.npz   四个因素的恢复质量及审计预测
    reference.npz            模型预测的参考轨迹与方差
    calibration.npz          校准窗口预测、评分与分差
    test_prediction.npz      所有 TEST 窗口的因素预测轨迹
    test_scores.npz          评分、排序、错选、交叉项与误差能量
```

运行结束后，只需压缩并发回这个文件夹：

```bash
tar -czf fourroom_01_results.tar.gz -C results fourroom_01
```

不需要另外收集文件。`assets/` 是运行输入缓存：官方源码、权重，以及本次生成的原始图像；`.venv/` 是运行环境。分析结果时不需要把它们发回。失败时也可以用同一命令打包已有结果供诊断。

名称不能重复，脚本会拒绝覆盖已有目录。重跑可改用 `fourroom_02`，但固定配置会生成相同的数据与划分；改名不意味着增加了一个独立实验重复。

## 3. 这次具体检验什么

使用官方 FourRoom 两层 H-JEPA 的三个模型，训练种子为 42、43、44。图像包含可控角色与一个干扰物。四个候选因素预先固定为角色 x、角色 y、干扰物 x、干扰物 y；每个窗口选出其中两个变化最慢的因素。不会看过结果后删掉难恢复的因素。

每层都从同一时刻的 RGB 图像产生表征，采用官方 dense probing 的逐帧编码方式。上层不跳帧，因此两层的 4、16、32 步滞后对应相同的物理时间。编码器不接收坐标、动作或未来帧。每层使用相同规格的两层 128 维 SiLU 隐藏层读出，优化 3,000 步；只用 TRAIN 拟合标准化和权重，只用 VAL 选读出训练中的最优检查点。

FourRoom 以官方 `fourroom_tp35_d1.yaml` 为基础，新采样种子为 2026111541，保持每段 400 次动作、401 帧；不使用官方存在重叠的训练/探测数据子集。另有一处明确的评价策略调整：把 expert policy 的 `action_repeat_prob` 从 0.05 固定为 0。原因是官方实现没有在每段重置时清除上次动作，重复动作可能把上一段的状态带入新 episode；关闭这项机制后再按独立 episode 统计。它改变了这一项动作采样设置，因此结果不宣称精确复现官方原评价分布。

按整段 episode 划分，六种用途互不重叠：

| 用途 | episode 数 | 如何使用 |
|---|---:|---|
| TRAIN | 256 | 每段固定抽 32 帧，共 8,192 个有标签状态，拟合读出 |
| VAL | 64 | 每段固定抽 32 帧，共 2,048 个状态，选读出检查点 |
| AUDIT | 128 | 每段一个 193 帧窗口，检查四个因素的恢复质量 |
| REF | 256 | 每段一个 193 帧窗口，固定真实/预测轨迹各自的参考方差 |
| CAL | 1,024 | 每段一个 193 帧窗口，只用预测分差定阈值 |
| TEST | 2,048 | 每段一个 193 帧窗口，最后检验选取是否正确 |

这里的读出预算是预先固定的视觉实验设置，并非与低维 LeJEPA 的所有拟合细节逐项相同。AUDIT 要求四个因素的 R² 都至少为 0.85；这是本轮操作性筛选条件，不是已经证明的普遍充分条件。未通过只说明这个读出没有验证因素可恢复，不能据此认定编码器已经丢失了因素信息，也不等同于否定可辨识性定理。

## 4. 评分和通过条件

对于每个因素，在一个 193 帧窗口中计算 4、16、32 步平方增量的平均值，各滞后等权。真实因素评分除以共同 REF 轨迹上的真实方差；读出因素评分除以该读出在同一组 REF 状态上的预测方差。FourRoom 的总体方差没有已知解析值，因此这里明确以独立 REF 定义物理评分，不把它假设为 1，也不在每个窗口内重新标准化。

分别选真实评分和预测评分最低的两个因素，集合不同即为一次错选。预测第 3 名减第 2 名的评分是可见分差。分差达到阈值才作答，否则暂不选取。精确平局按固定因素顺序处理；平局窗口保留。

每个模型检查两条预先指定的规则：

1. `own_cal`：每层用自身 CAL 预测分差的中位数作为阈值。
2. `level1_cal`：把该模型第 1 层的 CAL 阈值原样用于两层，检验跨层直接沿用阈值的边界。第 1 层两条规则相同，仍如实保留。

所有读出、参考方差和阈值先固定，再生成 TEST 预测；所有 TEST 预测保存并记录哈希后，才读取 TEST 坐标进行评分。不会用 TEST 调阈值，也不会把低维实验的旧阈值直接当作视觉实验保证。

正式实验共 3 个模型 × 2 层 × 2 条规则，按 12 项同时比较分配单侧 95% 置信预算。每项使用独立 episode 窗口上的 Clopper–Pearson 错选率上界。不同层和模型可以共用 TEST episodes；Bonferroni 校正不要求不同项彼此独立。每项内部把各新 episode 视为来自固定生成过程的独立样本，不把同段轨迹的帧当作独立样本。

`PASS` 必须同时满足：恢复审计通过、测试集作答比例至少 35%、校正后的作答错选率上界不超过 1%。覆盖率要求指样本观测比例，不是对总体覆盖率的置信保证。零观察错选也会报告非零统计上界。

这些结果只回答固定官方模型、FourRoom 生成过程、读出和评分设置下是否可靠。H-JEPA 学到的动作相关信息、表征是否可恢复、物理因素是否慢，是不同问题；不会用“上层表征变化较慢”代替物理因素选取验证。

## 5. 本地验证与 GPU

本地验证情况记录在 `VALIDATION.md`。完整服务器实验尚未运行，不能把代码检查或工程冒烟结果当成新的科学结论。

已有的低维 LeJEPA 实验不需要为了换 GPU 而重跑，GPU 对它们主要提供加速。这一轮图像版 H-JEPA 推荐用 GPU；如果未来训练图像版 LeJEPA，也建议使用 GPU。

## 官方来源

- [H-JEPA 官方代码](https://github.com/kevinghst/H-JEPA)，固定 commit `97054ecaff83e13839195ec4b92d74c08a0127c5`。
- [H-JEPA 官方模型](https://huggingface.co/jepa-world-models/h-jepa)，固定 revision `7b8ba64389690345045650c51c6d6fea0ba4bd02`。
- [H-JEPA 论文](https://arxiv.org/abs/2610.06805)。这里的 H-JEPA 指 Hierarchical JEPA。
