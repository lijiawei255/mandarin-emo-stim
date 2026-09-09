# 仓库审核改进 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/superpowers/specs/2026-09-10-repo-audit-design.md` 落实五个阶段：代码修复、UI 主题与视觉验证、文档科学性修正、三层验证评测、仓库规范与 Release。

**Architecture:** 保持现有分层（audio / models / features / fusion / stimulus / gui / storage）不变；新增 `src/gui/theme.py`（唯一调色板来源）、`scripts/ui_geometry_check.py`（布局几何检查）、`scripts/evaluate.py`（三层验证）、`config/prosody_norms.json`（实测基准）。每个任务先写失败测试，再实现，再提交。

**Tech Stack:** Python 3.10, PySide6 6.7, pytest, ruff, GitHub Actions, HuggingFace `datasets`（仅评测可选依赖）。

**验证命令**（环境 `conda run -n mandarin-emo-stim`）：
- 单元：`python -m pytest -q -p no:cacheprovider`
- GUI：`QT_QPA_PLATFORM=offscreen python -m pytest -q -m slow`
- GPU：`python -m pytest -q -m gpu`

---

## 阶段 A：代码修复与方法学实现改进

### Task A1: 管线降级不再崩溃
**Files:** Modify `src/pipeline.py`, `src/cli.py`, `src/gui/main_window.py`；Test `tests/test_pipeline_degradation.py`（新建）
- [ ] 测试：注入一个 `get_emotion_model()` 抛异常的假 manager，`analyze()` 必须返回结果且 `result["degraded_modalities"] == ["acoustic"]`，且 `negative` 在 [0,1]。
- [ ] 实现：`_step_*` 失败时写入中性分 (0.5,0.5) 并追加到 `ctx["degraded"]`；`_finalize` 输出 `degraded_modalities`。ASR 失败时文本两模态降级而非抛出（仅「加载音频」为致命）。
- [ ] `src/cli.py` 打印降级列表；`main_window._on_analysis_done` 在警告标签显示降级提示。
- [ ] 跑测试，提交 `fix(pipeline): 非关键模态失败时降级而非 KeyError`。

### Task A2: Ctrl+C 定时器接 Python 槽
**Files:** Modify `app.py`；Test `tests/test_robustness.py`
- [ ] 测试：`Application._make_sigint_timer(app)` 返回的 QTimer 触发后计数器递增（连接了 Python 槽）。
- [ ] 实现：`timer.timeout.connect(lambda: None)` 抽成 `_make_sigint_timer`。
- [ ] 提交 `fix(app): SIGINT 定时器连接 Python 槽，使 Ctrl+C 真正生效`。

### Task A3: PANNs 标签路径与 DataParallel
**Files:** Modify `src/models/pann_model.py`, `src/models/downloader.py`
- [ ] 测试（`tests/test_robustness.py`）：`PANNModel._load_labels()` 优先从 `portable.PANNS_DIR/class_labels_indices.csv` 读取；缺失时 `detect()` 返回 `degraded=True`。
- [ ] 实现：下载器新增 `download_panns_labels()`（URL `https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv`，随 `ensure_all_models` 一并保证）；`_load_labels` 先查 portable 目录，再回退 `~/panns_data`；移除 `DataParallel`；`detect` 返回值加 `degraded` 字段并在标签缺失时 `logger.warning`。
- [ ] 提交 `fix(panns): 标签落到 portable_data，缺失显式降级，移除单卡 DataParallel`。

### Task A4: 语速改用 ASR 时间戳，f0_drop 改为 F0 斜率，HNR 过滤修正
**Files:** Modify `src/features/prosody.py`, `src/audio/vad.py`, `src/pipeline.py`, `src/fusion/normalizer.py`；Test `tests/test_audio_features.py`
- [ ] 测试：`vad.syllable_rate(asr_result)`：3 字、时间戳覆盖 0.6 s → 5.0；`prosody.extract(y, sr, syllable_rate=5.0)` 时 `speech_rate == 5.0`；合成 F0 下滑信号的 `f0_slope < 0`，上升信号 `> 0`。
- [ ] 实现：`ProsodyFeatures` 新增 `f0_slope`（Hz/s，有声帧 F0 对时间线性回归）；`extract(..., syllable_rate=None)`；`score()` 中 `n_f0_drop = 1 - zscore(f0_slope, *stats["f0_slope"])`；`PROSODY_STATS` 加 `"f0_slope": (0.0, 40.0)` 占位（阶段 D 实测替换）；HNR 过滤改为排除 parselmouth 无定义哨兵（-200）。
- [ ] `pipeline._step_prosody` 传 `syllable_rate`。
- [ ] 提交 `feat(prosody): 语速用 ASR 时间戳、f0_drop 用真实斜率、HNR 保留负值帧`。

### Task A5: LLM greedy 优先
**Files:** Modify `src/models/llm_model.py`；Test `tests/test_robustness.py`
- [ ] 测试：注入假 tokenizer/model，记录两次 `generate` 的 `do_sample`：首次 False，重试 True。
- [ ] 实现：首次 `temperature=0.0`（greedy），失败重试 `0.3`。
- [ ] 提交 `fix(llm): 首次调用 greedy 解码，保证可复现`。

### Task A6: 合成器 RMS 归一 + 真正的峰值限幅
**Files:** Modify `src/stimulus/synthesizer.py`, `src/stimulus/generator.py`；Test `tests/test_stimulus.py`
- [ ] 测试：输出峰值 ≤ `10**(max_peak_dbfs/20)`；loud_db=-10 时峰值被限幅到 -10 dBFS（误差 1e-3）；loud_db 越大 RMS 越大。
- [ ] 实现：第 6 步按 RMS 归一，第 7 步限幅读取 `max_peak_dbfs`（generator 把 settings.stimulus 合并进 stimulus_config），删除末尾 `*0.7`。
- [ ] 提交 `fix(synth): 响度按 RMS 归一，峰值限幅真正生效`。

### Task A7: 死代码清理与面板重命名
**Files:** Modify `src/gui/main_window.py`, `src/gui/threads.py`, `src/gui/styles.qss`
- [ ] 删除 `ModelLoadWorker`、`_on_model_progress`、`_on_models_loaded`、`_connect_workers_slots`、`model_worker`；docstring 改为「模型加载在主线程分阶段」。
- [ ] 对象名 → `InputPanel/MetricsPanel/ModalPanel/StimulusPanel`；方法名同步。
- [ ] `pytest -m slow` 通过；提交 `refactor(gui): 移除死代码，面板语义化命名`。

### Task A8: 科学行为回归测试
**Files:** Create `tests/test_scientific_behavior.py`
- [ ] 合成信号断言：F0 300 vs 150 → `a_prosody` 更高；加噪（HNR 低）→ `s_prosody` 更高；双音 70 Hz 拍频的 roughness 高于纯音；Q2 高 arousal 脉冲率 < 低 arousal，Q3 相反；象限角点归属正确。
- [ ] 提交 `test: 新增科学行为回归测试`。

## 阶段 B：UI 主题与视觉验证

### Task B1: `src/gui/theme.py` 调色板 + QSS 生成
- [ ] 新建 `theme.py`：`PALETTE`（bg #FAF9F5、card #FFFFFF、card_alt #F0EEE6、border #E5E2D9、text #1F1E1D、text_muted #6E6D68、accent #D97757、accent_dark #C4633F、success #788C5D、warning #B8860B、error #BC4C3C、radius 8）；`build_qss()` 读取 `styles.qss.tmpl` 做 `{{key}}` 替换；`app.py` 与 `scripts/gui_screenshot.py` 改用 `build_qss()`。
- [ ] 5 个控件与 `main_window.py` 内联颜色全部改为引用 `PALETTE`。
- [ ] 测试：`build_qss()` 不含 `{{`，包含 `#D97757`；`src/gui/**` 除 theme.py 外不出现十六进制色值。
- [ ] 提交 `feat(ui): 暖奶油主题，调色板单一来源`。

### Task B2: 几何检查脚本与测试
- [ ] 新建 `scripts/ui_geometry_check.py`：`collect_issues(window)`，检查 (a) 同父可见兄弟控件矩形相交；(b) 非 wordWrap 的 QLabel/QPushButton `sizeHint().width() > width()`；(c) 子控件超出父 rect；(d) `height() < minimumSizeHint().height()`。
- [ ] 新建 `tests/test_gui_layout.py`（slow）：3 分辨率 × 3 状态，`assert issues == []`。
- [ ] 提交 `test(ui): 布局几何自动检查`。

### Task B3: 截图 + 视觉模型复核
- [ ] `scripts/gui_screenshot.py` 输出 9 张到 `docs/images/ui/`（真实平台渲染）。
- [ ] 用 qwen3-vl-plus 逐张判读，记录到 `docs/superpowers/plans/ui-visual-check.md`。
- [ ] 提交 `docs(ui): 新主题截图与视觉复核记录`。

## 阶段 C：文档科学性修正
### Task C1: `docs/research_notes.md`
- [ ] 锚点补引用并标注启发式；语速/f0_slope 更新；ASR 置信度代理说明；补 Chinese EmoBank/CVAW；删除粉噪平复主张；新增 ISO 原则讨论；修正 SPL 与基准值来源；局限性补充；参考文献更新。
### Task C2: README（中英）、`docs/user_guide.md`、`docs/faq.md`、`main_window.on_about`
- [ ] 安全声明统一改为「数字峰值限幅，实际声压取决于设备」；界面描述改为新主题；「验证状态」小节。
- [ ] 提交 `docs: 修正引用错配、SPL 声明、界面描述`。

## 阶段 D：三层验证
### Task D1: `scripts/evaluate.py` 骨架 + `requirements-eval.txt`
- [ ] 子命令 `calibrate`、`neutral`、`emotion`、`report`；数据经 `datasets` 读取，缓存到 `portable_data/eval/`。
### Task D2: `calibrate` → `config/prosody_norms.json`，`normalizer.py` 改读该文件（缺失回退常数）。
### Task D3: `neutral`：CER + 输出分布。
### Task D4: `emotion`：混淆矩阵、方向一致性、Spearman、模态消融、动态权重开关。
### Task D5: 运行并写 `docs/evaluation.md`。
- [ ] 提交 `feat(eval): 三层验证脚本与结果`。

## 阶段 E：仓库规范与 Release
### Task E1: `pyproject.toml`、`ruff check` 清零。
### Task E2: `.github/workflows/ci.yml`。
### Task E3: `CITATION.cff`、`CODE_OF_CONDUCT.md`、issue/PR 模板、`.gitignore` 精简、`CHANGELOG.md` 0.1.0。
### Task E4: README 徽章、截图、Mermaid 架构图、验证状态；仓库 description/topics。
### Task E5: push、tag v0.1.0、Release 附截图与四象限刺激样例。
