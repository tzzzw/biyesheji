# 基于 YOLO 的无烟区域吸烟者实时检测与智能预警系统

本项目是一个最小可运行的毕业设计代码骨架，基于 `Python + Ultralytics + OpenCV`，实现以下流程：

- 检查数据集结构与 `data.yaml`
- 将混合标注数据转换为运行时检测数据集
- 训练并对比 `YOLOv8` 与 `YOLO26`
- 评估训练结果
- 进行摄像头或视频的实时检测与禁烟区预警

当前项目代码位于 [smoke_project](/d:/毕设/smoke_project:1)。

## 1. 项目结构

```text
smoke_project/
├─ check_dataset.py
├─ search_train_config.py
├─ mainline_converge.py
├─ freeze_baseline.py
├─ error_analysis.py
├─ tune_small_budget.py
├─ train_compare.py
├─ evaluate.py
├─ realtime_alarm.py
├─ README.md
├─ generated_data/
└─ runs/
```

数据集目录位于：

```text
D:\毕设\Smoking_yolov8
```

注意：当前项目会优先使用 `Smoking_yolov8`，如果该目录不存在，再兼容回退到旧目录 `cigarette_smoker_yolov8`。

## 2. 环境说明

你当前已经可用的环境是：

```text
C:\Users\tzw12\anaconda3\envs\yolo\python.exe
```

如果 VS Code 里提示 `ultralytics` 未安装，通常是解释器选错了。请将 VS Code 解释器切换到上面的 `yolo` 环境。

建议安装命令：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe -m pip install ultralytics opencv-python pyyaml
```

## 3. 数据集说明

原始数据集是单类目标检测数据集，类别为：

```yaml
names:
  0: smoking
```

但当前原始数据集存在两个实际问题：

- `data.yaml` 的路径写法使用了 `../train/images` 这一类相对路径，不同脚本直接解析时不够稳定
- 原始标签中可能存在极少量无效框，本项目会在运行时数据集生成阶段自动跳过无效标注行

说明：项目内部会把 `smoking` 视作与 `smoker` 兼容的单类标签，并统一生成规范的运行时 `data.yaml`。

因此本项目不会直接修改原始数据集，而是在训练前自动生成一个“运行时检测数据集”：

```text
smoke_project/generated_data/smoker_detection_runtime
```

这个运行时数据集会由当前主线脚本 [search_train_config.py](/d:/毕设/smoke_project/search_train_config.py:1) 自动复用生成逻辑；底层准备函数仍然定义在 [train_compare.py](/d:/毕设/smoke_project/train_compare.py:1) 中。

## 4. 各脚本功能

- [check_dataset.py](/d:/毕设/smoke_project/check_dataset.py:1)
  只读检查数据集结构、标签格式、`data.yaml` 配置
- [search_train_config.py](/d:/毕设/smoke_project/search_train_config.py:1)
  当前正式实验主线脚本，用于固定配置冒烟、资源验证、正式验证和可恢复长训
- [mainline_converge.py](/mnt/毕设/smoke_project/mainline_converge.py:1)
  当前“主线模型收敛训练阶段”脚本：保留 YOLOv8s 基线结果，只续训 YOLO26s，并自动输出论文汇总表
- [freeze_baseline.py](/mnt/毕设/smoke_project/freeze_baseline.py:1)
  将当前选定的 `best.pt`、`results.csv` 和现有评估目录冻结到独立 baseline 目录，并生成 `baseline_summary.md`
- [error_analysis.py](/mnt/毕设/smoke_project/error_analysis.py:1)
  对冻结基线或指定 `best.pt` 做误差分析，导出误检样本、漏检样本、典型失败案例和可视化结果
- [tune_small_budget.py](/mnt/毕设/smoke_project/tune_small_budget.py:1)
  基于冻结基线做小预算超参数调优，默认按 `optimizer/lr0 -> weight_decay/warmup_epochs -> mosaic -> confirmation` 顺序推进，并输出主正式训练参数与备选参数
- [train_compare.py](/d:/毕设/smoke_project/train_compare.py:1)
  辅助对比脚本，用于早期联调、双模型快速对比和历史兼容；不再作为当前 `s` 级正式主线入口
- [evaluate.py](/d:/毕设/smoke_project/evaluate.py:1)
  自动读取最新训练结果并做评估
- [realtime_alarm.py](/d:/毕设/smoke_project/realtime_alarm.py:1)
  调用最新训练权重进行实时检测和禁烟区预警

## 5. 推荐运行顺序

### 第一步：检查数据集

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/check_dataset.py
```

### 第二步：生成运行时数据集

如果你只是想提前准备运行时数据集，可以继续用辅助脚本：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/train_compare.py --prepare-only --force-rebuild
```

如果你准备直接进入当前正式实验主线，通常不需要单独做这一步，`search_train_config.py` 会自动复用相同的数据准备逻辑。

### 第三步：做一次最小 CPU 训练联调

如果没有显卡，先只跑一个模型验证流程。这里继续使用辅助脚本即可：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/train_compare.py --cpu-friendly --skip-model-b --model-a yolov8n.pt --label-a yolov8_cpu_test --epochs 1 --batch 2 --imgsz 320
```

说明：

- `--cpu-friendly` 会自动降低资源消耗
- `--skip-model-b` 先不跑第二个模型
- `yolov8n.pt` 第一次运行时一般会自动下载

### 第四步：正式做模型对比

当前正式实验主线推荐使用 `search_train_config.py`，按“单模型固定配置”运行。

例如，对 `YOLOv8s` 做一次固定配置验证：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/search_train_config.py --formal-verify --model yolov8s.pt --imgsz-values 896 --batch-values 48 --epochs 30 --workers 16 --device 0 --seed 42 --low-prefetch
```

如果你只是想做早期双模型快速对比，仍然可以使用 `train_compare.py`：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/train_compare.py --cpu-friendly --model-a yolov8n.pt --label-a yolov8 --model-b 你的_yolo26_模型文件 --label-b yolo26 --epochs 30
```

如果 `YOLO26` 的模型名、权重文件名或训练方式不同，请根据实际情况替换 `--model-b`。

### 第五步：评估模型

默认会自动读取最新一次训练输出：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/evaluate.py
```

如果只想评估指定权重：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/evaluate.py --weights d:/毕设/smoke_project/runs/compare_xxx/yolov8_cpu_test/weights/best.pt --labels yolov8_cpu_test
```

### 第五步补充：进入主线模型收敛训练阶段

如果你已经完成 `YOLOv8s` 与 `YOLO26s` 的同条件 30 epoch 筛选，并决定后续只保留 `YOLO26s` 作为主线模型，可以直接运行：

```bash
python /mnt/毕设/smoke_project/mainline_converge.py --target-epochs 100 --device 0
```

该脚本会自动完成：

- 保留当前 `YOLOv8s` 30 epoch 结果，作为论文基线对比结果
- 保留当前 `YOLO26s` 30 epoch 结果，作为模型筛选依据
- 仅对 `YOLO26s` 执行 `resume` 续训到目标轮数
- 长训完成后自动执行 `val/test` 正式评估
- 输出论文可用的：
  - `model_selection_table.md`
  - `model_selection_comparison.csv`
  - `yolo26s_final_eval.md`
  - `yolo26s_final_evaluation.csv`
  - `summary.md`

注意：

- 不建议直接使用 `YOLO(last.pt).train(resume=True, epochs=100)` 这类简写方式做正式续训
- 在当前环境的 `Ultralytics 8.4.41` 中，这种写法以及部分 CLI 写法都可能不会真正读取原训练参数，而是错误地新开一个默认训练任务
- 一旦你看到新目录落到 `runs/detect/train` 或 `runs/detect/train-*`，并且 `args.yaml` 里出现 `data: coco8.yaml`、`resume: false`，就说明这不是原实验的正式续训
- 当前项目里的 [mainline_converge.py](/mnt/毕设/smoke_project/mainline_converge.py:1) 会优先寻找“仍保留 optimizer/EMA 状态”的可恢复 checkpoint
- 对于当前环境里 `optimizer=auto` 与旧 checkpoint 优化器状态不兼容的问题，脚本会自动把 resume 源里的优化器类型固定为 checkpoint 对应类型，避免 `loaded state dict has a different number of parameter groups`
- 如果最终 `last.pt/best.pt` 已被 Ultralytics strip 成仅推理权重，脚本会自动回退到最近一个可恢复的 `epoch*.pt`，并把长训结果写入新的独立目录，避免污染 30 epoch 模型筛选结果
- 脚本还会在启动前清理上次失败遗留的 `mainline_converge.py` 孤儿进程，并把 `workers` 自动压到当前主机更稳的范围，减少被系统直接 `Killed` 的概率
- 如果首轮反向传播仍出现 `CUDA out of memory`、`CUBLAS_STATUS_ALLOC_FAILED` 一类资源错误，脚本会自动降低 `batch` 并重试，直到找到当前机器可稳定启动的配置

### 第五步补充：基线冻结与误差分析阶段

如果你当前已经不准备继续直接长训，而是先冻结现有基线并进入提升方案，可以先执行：

```bash
python /mnt/毕设/smoke_project/freeze_baseline.py
```

默认行为：

- 自动读取 `smoke_project/runs` 下最新的 `search_summary.json`
- 优先冻结推荐结果或该摘要里指标最优的成功结果
- 复制对应 `best.pt`、`results.csv`、现有 `validation` 目录
- 在新的 `baseline_freeze_<timestamp>` 目录下生成：
  - `baseline_summary.md`
  - `baseline_manifest.json`

如果你想显式指定来源，也可以传：

```bash
python /mnt/毕设/smoke_project/freeze_baseline.py \
  --source-run-dir /mnt/毕设/smoke_project/runs/你的_run_dir
```

冻结完成后，可直接对该基线执行误差分析：

```bash
python /mnt/毕设/smoke_project/error_analysis.py --split test --device 0
```

默认行为：

- 自动读取最新的 `baseline_manifest.json`
- 复用运行时检测数据集
- 在 `test` 划分上分析误检、漏检和混合失败样本
- 导出：
  - `image_summary.csv`
  - `fp_images.csv`
  - `fn_images.csv`
  - `fp_boxes.csv`
  - `fn_boxes.csv`
  - `typical_failures.md`
  - `visuals/`

如果你只是先检查流程是否正常，也可以限制分析样本数：

```bash
python /mnt/毕设/smoke_project/error_analysis.py --split test --max-images 32 --device 0
```

冻结基线并完成误差分析后，可以进入小预算超参数调优。先预览默认第一轮实验计划：

```bash
python /mnt/毕设/smoke_project/tune_small_budget.py --plan-only
```

默认第一轮会固定：

- `imgsz=896`
- `batch=48`
- `workers=16`
- `weight_decay=0.0005`
- `warmup_epochs=3.0`
- `mosaic=1.0`

并只测试 4 组：

- `AdamW + lr0=0.005`
- `AdamW + lr0=0.01`
- `SGD + lr0=0.005`
- `SGD + lr0=0.01`

如果准备直接启动第一轮小预算调参：

```bash
python /mnt/毕设/smoke_project/tune_small_budget.py --rounds 1 --device 0
```

如果第一轮有结果后想继续到第二轮或第三轮，可以在同一个输出目录上继续：

```bash
python /mnt/毕设/smoke_project/tune_small_budget.py \
  --project /mnt/毕设/smoke_project/runs/你的_hparam_budget_目录 \
  --rounds 2 --device 0
```

```bash
python /mnt/毕设/smoke_project/tune_small_budget.py \
  --project /mnt/毕设/smoke_project/runs/你的_hparam_budget_目录 \
  --rounds 3 --device 0
```

说明：

- 所有小预算 trial 都从同一个冻结基线 `best.pt` 启动
- 后续轮次只继承前一轮的“优胜超参数”，不会继承前一轮训练出来的权重
- 这样可以保证每组试验的起点一致，方便公平比较
- 当 `--rounds 3` 且保持默认 `--confirmation` 时，脚本会在第三轮 `mosaic` 搜索结束后自动进入确认实验阶段
- 确认实验会：
  - 对第三轮当前最优组用不同 `seed` 再跑一次
  - 同时对保留的备选组再跑一次对照实验
- 默认保留的备选组为：
  - `optimizer=SGD`
  - `lr0=0.005`
  - `weight_decay=0.001`
  - `warmup_epochs=5`
  - `mosaic=第三轮当前最优值`
- 默认确认实验参数为：
  - `seed=123`
  - `epochs=24`
  - `patience=10`
- 确认阶段结束后会额外输出：
  - `confirmation_summary.md`
  - `final_recommendation.md`
- `final_recommendation.md` 会明确给出：
  - 主正式训练参数
  - 备选参数
  - 基于 `recall` 与 `mAP50-95` 权衡的推荐理由

### 第六步：实时检测与智能预警

使用默认摄像头：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0
```

使用本地视频文件：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 你的视频文件.mp4 --save-video
```

自定义禁烟区：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --zone 100,100 500,100 500,400 100,400
```

进入第二阶段交互式绘制禁烟区：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --draw-zone
```

绘制并保存区域配置：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --draw-zone --zone-config d:/毕设/smoke_project/configs/lab_zone.json
```

加载已保存的区域配置：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --zone-config d:/毕设/smoke_project/configs/lab_zone.json
```

启用第一阶段“时序智能预警”增强参数：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --alarm-frames 6 --alarm-avg-conf 0.55 --alarm-cooldown 5
```

启用第三阶段“违规事件闭环”并保存报警前后短视频：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --event-pre-seconds 2 --event-post-seconds 3
```

如果只想保留截图和结构化日志，不保存事件短视频：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/realtime_alarm.py --source 0 --no-save-event-clip
```

当前实时预警已支持以下状态机状态：

- `IDLE`：当前没有区内吸烟目标
- `OBSERVING`：正在累计连续命中帧和平均置信度
- `ALARMED`：满足触发条件，本帧执行报警
- `COOLDOWN`：报警后冷却，避免短时间重复触发

交互式绘制窗口操作说明：

- 鼠标左键：添加顶点
- 鼠标右键或 `u`：撤销上一个顶点
- `c`：清空当前多边形
- `Enter` 或空格：确认当前区域
- `Esc` 或 `q`：取消绘制

退出按键：

```text
q
```

## 6. 输出目录说明

### `generated_data/`

用于保存自动生成的运行时检测数据集，例如：

- `smoker_detection_runtime/data_runtime.yaml`
- `smoker_detection_runtime/prepare_summary.json`

### `runs/`

用于保存训练、评估和实时检测结果，例如：

- `formal_verify_*/summary.md`
- `formal_verify_*/result.csv`
- `search_*/search_summary.csv`
- `compare_*/comparison_summary.json`
- `compare_*/模型名/weights/best.pt`
- `evaluate_*/evaluation_summary.json`
- `realtime_*/alarm_events.jsonl`
- `realtime_*/alarm_events.csv`
- `realtime_*/realtime_result.mp4`
- `realtime_*/events/event_0001/snapshot.jpg`
- `realtime_*/events/event_0001/clip.mp4`
- `realtime_*/events/event_0001/event.json`

其中 `alarm_events.jsonl` 在第一阶段会记录更完整的时序预警字段，例如：

- `event_id`
- `state`
- `hit_frames`
- `frame_average_confidence`
- `streak_average_confidence`
- `best_confidence`
- `snapshot_path`
- `clip_path`

第二阶段保存的区域配置文件由你通过 `--zone-config` 或 `--save-zone-config` 指定，例如：

- `configs/lab_zone.json`
- `configs/dorm_zone.json`

## 7. 当前已完成内容

目前这个最小项目已经具备：

- 数据集检查
- 运行时检测数据集生成
- 最小训练流程
- 模型评估脚本
- 实时报警脚本
- 时序智能预警第一阶段
  - 连续多帧命中
  - 平均置信度约束
  - 冷却时间抑制
  - 可视化状态机
- 交互式禁烟区域第二阶段
  - 鼠标绘制多边形区域
  - 区域配置保存为 `json`
  - 区域配置加载复用
  - 与现有 `--zone` 参数兼容
- 违规事件闭环第三阶段
  - 报警截图自动保存
  - 报警前后短视频自动导出
  - 事件 `jsonl/csv/json` 结构化日志
  - 单个事件按 `event_id` 归档管理

你已经完成过 `n` 级正式验证和 `s` 级冒烟迁移，当前仍保留的代表性结果包括：

- [summary.md](/mnt/毕设/smoke_project/runs/formal_verify_20260424_e30_img896_b48_w16/summary.md:1)
- [summary.md](/mnt/毕设/smoke_project/runs/s_model_migration_20260424/smoke_yolov8s_e5_img896_b48_w16/summary.md:1)
- [summary.md](/mnt/毕设/smoke_project/runs/s_model_migration_20260424/smoke_yolo26s_e5_img896_b48_w16/summary.md:1)

## 8. 常见问题

### 1. 明明安装了 `ultralytics`，为什么还提示没安装？

原因通常是 Python 环境不一致。

当前默认 `python` 可能是：

```text
C:\Users\tzw12\anaconda3\python.exe
```

而真正安装 `ultralytics` 的环境是：

```text
C:\Users\tzw12\anaconda3\envs\yolo\python.exe
```

所以请尽量始终使用：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe
```

### 2. 为什么 `check_dataset.py` 会报错？

这是因为它在帮你识别原始数据集里的真实问题：

- `data.yaml` 路径设置不稳定
- 标签里混有检测框和分割标注

这不是脚本写错，而是检查成功了。

### 3. 没有显卡还能做吗？

可以。

建议策略：

- 本地电脑先完成“流程跑通”
- 正式实验主线优先使用 `search_train_config.py`
- 双模型快速联调再使用 `train_compare.py`

本地 CPU 最推荐的测试命令是：

```bash
C:/Users/tzw12/anaconda3/envs/yolo/python.exe d:/毕设/smoke_project/train_compare.py --cpu-friendly --skip-model-b --model-a yolov8n.pt --label-a yolov8_cpu_test --epochs 1 --batch 2 --imgsz 320
```

## 9. 后续可扩展方向

- 支持多禁烟区和区域命名管理
- 支持多摄像头接入
- 增加语音播报或弹窗提醒
- 接入数据库或 Web 后台
- 进一步补充 `YOLO26` 的正式对比实验
