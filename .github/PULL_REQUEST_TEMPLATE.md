## 变更内容 / What

<!-- 一两句话说明改了什么、为什么 -->

## 类型 / Type

- [ ] 修复 bug
- [ ] 新功能
- [ ] 方法学改动（特征 / 权重 / 映射 / 模型）
- [ ] 文档
- [ ] 重构 / 测试 / CI

## 检查清单 / Checklist

- [ ] `python -m pytest -q` 通过
- [ ] 涉及 GUI 时 `python -m pytest -q -m slow` 通过（含布局几何检查）
- [ ] 涉及方法学时：已在 `docs/research_notes.md` 标注证据等级（[文献] / [启发式] / [实测]）并给出出处
- [ ] 涉及方法学时：`tests/test_scientific_behavior.py` 仍通过，或已说明为何需要改动
- [ ] 未提交模型文件、录音、历史记录或任何隐私数据（`portable_data/` 应保持被忽略）
- [ ] 新增依赖的许可证与 Apache-2.0 兼容
