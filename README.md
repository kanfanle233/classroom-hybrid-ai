# YOLOv11 Classroom Behavior Pipeline

课堂行为多模态分析流水线，包含 YOLOv11 姿态/目标检测、行为融合、ASR、事件验证、时间线生成，以及 FastAPI 可视化服务。

本仓库的 GitHub 版本只保存可复现的代码、契约、测试、配置和小型标注协议。原始视频、模型权重和运行输出保留在本地，不作为 GitHub 版本的一部分。

## 1. 仓库边界

### GitHub 中保留

- `classroom_pipeline/`：当前主流程代码、阶段注册表、融合、契约、验证器和清理工具。
- `server/`：FastAPI 后端、前端模板、manifest/registry 和回归代码。
- `scripts/`：维护、检查和实验辅助脚本；旧主流程脚本已移动到 `paper_experiments/legacy_*` 归档目录。
- `tests/`：回归测试和小型 fixture，不包含原始课堂视频。
- `contracts/`、`paper_experiments/configs/`：接口契约、样例和实验配置。
- `paper_experiments/gold/`：schema、模板、协议和小型金标 JSONL；证据帧图片不进入云端。
- `paper_experiments/real_cases/`：案例索引和小型汇总，不包含原始视频和模型运行目录。
- `paper_experiments/official_yolo_finetune_compare/`：官方 YOLO 基线配置、检查脚本和小型报告。
- `README.md`、`.gitignore`、必要的 Markdown 指南。

### GitHub 中删除/不上传

| 路径 | 本地盘点（2026-07-10） | GitHub 策略 | 原因 |
|---|---:|---|---|
| `data/` | 30,614 个文件，约 12.48 GiB；其中 12,843 个 MP4、8,884 个 JPG | 只保留本地 | 原始课堂视频、图片和标注体积大，且需要单独确认数据授权 |
| `models/` 权重 | 3 个 `.pt`，约 224.62 MiB | 只保留本地 | 权重可下载或由训练生成；GitHub 不存二进制权重 |
| `output/` | 1,766 个文件，约 1.51 GiB；主要是 JSONL、音频和视频 | 只保留本地 | 运行产物可再生成，且大量文件与具体机器路径相关 |
| `runs/` | 训练/推理运行目录 | 只保留本地 | 训练曲线、缓存和 checkpoint 都是可再生产物 |
| `docs/` 大型资产 | 709 个文件，约 0.24 GiB；含视频、图表、CSV 和 JSONL | 只保留必要 Markdown 指南 | 论文图表、演示视频和导出数据不是运行所需源码 |
| `paper_experiments/run_logs/` | 实验日志和重试日志 | 从 GitHub 删除 | 日志可重新生成，噪声大且不应作为主结果输入 |
| `paper_experiments/gold/*/evidence/` | 证据帧和 contact sheet 图片 | 从 GitHub 删除，保留本地 | 体积大，也可能包含课堂画面；金标协议和 JSONL 仍保留 |
| `.ultralytics/`、`__pycache__/`、`.pytest_cache/` | 本地缓存 | 从 GitHub 删除 | 工具自动生成，不属于源码 |
| `.DS_Store`、`__MACOSX/`、`.idea/`、机器专用 `.vscode/` | 历史导入杂项 | 从 GitHub 删除 | 与项目运行无关，且包含本机路径/IDE 状态 |
| 旧论文包、重复报告、旧入口和重复模板 | 历史归档 | 从 GitHub 删除或移动到轻量归档 | 避免把同一资产的多个副本当作当前实现 |

清理 GitHub 只会删除远程版本；工作区中的 `data/`、`models/`、`output/` 和证据图片按本地文件保留。若需要发布这些资产，应使用有权限控制的对象存储或数据集平台，并在这里记录版本号、校验和与授权信息，而不是直接提交到 Git。

## 2. 本地资产如何恢复

### 2.1 YOLO 预训练权重

主流程默认解析以下两个本地文件：

```text
models/yolo11x.pt       # 当前本地约 109.33 MiB
models/yolo11n-pose.pt  # 当前本地约 5.97 MiB
```

Ultralytics 可以按模型名下载预训练权重。安装好 `ultralytics` 后运行：

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt'); YOLO('yolo11n-pose.pt')"
```

如果下载器把文件放在仓库根目录，请将它们移动到 `models/`。旧参数如 `--det_model yolo11x.pt` 仍会由 `classroom_pipeline/model_resolver.py` 解析到 `models/yolo11x.pt`。

行为检测权重 `official_yolo11s_detect_e150_v1_best.pt` 和 `case_yolo_train_best.pt` 不是本仓库可自动获得的预训练权重，需要从对应训练运行或受控模型存储中取得。没有行为检测权重时，`doctor` 会给出 warning，主流程会回退到姿态/规则动作路径。

### 2.2 数据集和输入视频

原始数据应放在 `data/`，输入视频可以使用任意本地路径；README 不伪造第三方数据集下载地址。当前工作区的 `data/` 混合了智慧课堂视频、图片/JSON 标注以及处理后的数据，具体来源和许可证必须以数据提供方的正式发布页为准。

建议恢复后的最小结构为：

```text
data/
└─ <dataset-or-case>/
   ├─ videos/          # 原始或授权输入视频
   ├─ images/          # 可选图片数据
   ├─ labels/          # 可选 YOLO/JSON 标注
   └─ dataset.yaml     # 训练数据集需要时提供
```

不要把带有学生画面的原始视频、证据帧或未经脱敏的标注重新推送到公开仓库。

### 2.3 输出和论文实验资产

运行输出写到 `output/`，训练输出写到 `runs/`。这些目录在当前工作区约占 1.51 GiB 和更大的训练缓存空间，可通过同一输入、模型版本和配置重新生成。`paper_experiments/real_cases/` 只保留索引和小型汇总；完整 case output、JSONL、音频和叠加视频仍在本地 `output/`。

## 3. 安装和快速检查

项目没有把某台机器的 Conda 路径写进依赖文件。建议使用 Python 3.10+ 虚拟环境，并按实际功能安装依赖；最小检查通常需要 `opencv-python` 和 `ultralytics`，服务端还需要 `fastapi`/`uvicorn`，测试需要 `pytest`。

```powershell
python -m pip install ultralytics opencv-python fastapi uvicorn pytest
python run_pipeline.py doctor
```

`doctor` 会检查主流程文件、Python 模块和本地模型。缺少原始视频时可以先不传 `--video`；缺少行为检测权重只会产生 warning。

## 4. 推荐入口

```powershell
# 检查代码、契约和模型
python run_pipeline.py doctor

# 对本地视频运行主流程
python run_pipeline.py run --video data/<dataset-or-case>/videos/demo1.mp4

# 校验一个运行目录
python run_pipeline.py validate --case-dir output/<case>

# 只生成清理清单，不删除任何内容
python run_pipeline.py clean --dry-run
```

主流程代码位于 `classroom_pipeline/`：

```text
classroom_pipeline/
├─ pipeline/       # 姿态、检测、跟踪、ASR、验证和时间线阶段
├─ fusion/         # 语义化、对象/动作融合和契约检查
├─ contracts/      # JSON/JSONL schema 与样例
├─ verifier/       # 双验证器训练、评估和推理
├─ judge/          # teacher/student 判别器实验模块
├─ profiles/       # 语义配置
└─ cli.py          # run / doctor / validate / clean
```

服务端入口是 `server/app.py`，前端模板位于 `server/templates/`。服务端读取的是本地 `output/` 和可选的本地文档资产，不要求把这些大文件提交到 GitHub。

## 5. 复现和结果边界

- `paper_experiments/gold/` 中的 sample/template 不能当作正式实验结果。
- 真实课堂数据、金标和证据图片必须先完成脱敏、授权和 held-out 划分。
- 运行结果必须能追溯到输入 case、模型文件、配置和代码提交；不要把“本地存在”写成“GitHub 已发布”。
- `paper_experiments/legacy_*` 仅用于迁移追溯，不是当前主入口。
- 需要发布论文图表时，只发布去重后的最终资产，并在单独的发布包中记录来源和校验和。

## 6. GitHub 仓库

```text
https://github.com/kanfanle233/yolov11-classroom-pages
```

当前合并目标是远程 `main`。所有云端清理都以本 README 的边界为准：代码和小型可复现输入进入 Git，原始数据、权重、输出、日志和缓存留在本地或受控存储。
