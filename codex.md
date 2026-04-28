# Codex Notes

## 2026-04-28

- 待处理问题：Streamlit 前端入口依赖从 `app` 根目录启动。
- 现象：从 `app/` 目录启动时，`ui_module/streamlit_app.py` 能正常加载；如果从 `app/ui_module/` 直接启动，部分相对路径资源会找错，`BERTManager` 可能误判本地模型缺失并尝试下载，随后在 `winget` 处失败。
- 当前约定：先按 `cd app && streamlit run ui_module/streamlit_app.py` 使用。
- 处理状态：已记录，当前先不修。
