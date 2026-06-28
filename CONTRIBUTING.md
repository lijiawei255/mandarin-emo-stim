# 贡献指南

感谢你对 Mandarin-EmoStim 的关注！欢迎通过以下方式参与贡献。

## 开发环境

```bash
conda create -n mandarin-emo-stim python=3.10.14 -y
conda activate mandarin-emo-stim
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

详见 [README.md](./README.md)。

## 提交规范

1. 从 `main` 拉取最新代码，新建分支开发：`git checkout -b feat/your-feature`。
2. 代码需通过测试：`pytest`。
3. 提交信息建议带前缀：`feat:` / `fix:` / `docs:` / `test:` / `refactor:`。
4. 开启 PR，描述改动内容与动机。

## 开源合规要点

本项目为公开开源项目，请贡献者特别注意：

- **不要提交模型文件、用户录音、历史记录或任何个人隐私数据**——这些都应位于 `portable_data/`（已被 gitignore 排除）。
- **不要在代码中硬编码个人路径**，一律使用相对路径或 `src/portable.py` 提供的常量。
- 新增依赖需确认协议与 Apache 2.0 兼容。
- 测试夹具（如音频）必须明确标注来源与协议（CC / 公共领域）。

## 代码风格

- Python 3.10，类型注解优先。
- 函数/类需有 docstring（中文）。
- 新增模块需配套测试（`tests/`）。

## 报告问题

请在 GitHub Issues 中描述问题，附上：
- 复现步骤；
- 运行环境（OS、显卡、Python 版本）；
- 相关日志（`portable_data/logs/`，注意去除隐私信息）。
