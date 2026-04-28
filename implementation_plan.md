# Streamlit 前端 Demo 实施计划

通过这个简单的 Streamlit 应用，我们可以将现有的 `BERT分类 -> 推荐系统召回排序 -> DeepSeek大模型包装` 完整主链路进行可视化展示。

## 需求理解与环境说明
1. 目标：创建一个独立的 Streamlit Web UI (`app/ui_module/streamlit_app.py`)，用于接收病情描述，并以精美的 JSON 卡片形式展示 DeepSeek 的诊断和推荐结果。
2. 关于 conda 7606：后续启动应用时，建议在名为 `7606` 的 conda 环境中运行（或指定端口 7606），命令如 `conda run -n 7606 streamlit run ui_module/streamlit_app.py --server.port 7606`。

## User Review Required

> [!IMPORTANT]
> 由于原来的 `DeepSeekManager` 采用了**队列+后台异步线程**的方式（非阻塞式），而在 Web 端我们通常需要**同步等待** DeepSeek 的返回结果渲染网页。
> 
> 我的解决方案是：直接在 Streamlit 中复用单例 `DeepSeekManager` 的内置参数（如 `client` 和 `_prompt_build()` 模板引擎），然后在 Streamlit 主线程发起一个**同步阻塞的请求**等待大模型回复。这样不用去大改你原有的异步队列代码。请问这个方案可以吗？

## Proposed Changes

### ui_module
这里将新增一个专门用于 Web Demo 的 Streamlit 入口脚本。

#### [NEW] [streamlit_app.py](file:///Users/jayden/Desktop/7012%20datamining%20and%20text/project_march/ARIN7102_Group_Project/app/ui_module/streamlit_app.py)
- **初始化单例**：加载 `BERTManager`、`DrugRecommendationService` 和 `DeepSeekManager`。
- **页面布局**：包含标题、侧边栏配置说明、以及一个巨大的症状描述输入框 (`st.text_area`)。
- **核心逻辑**：
  1. 用户点击“开始诊断”按钮。
  2. 获取 `bert_output` 和 `pipeline_output`。
  3. 调用 `deepseek_manager._prompt_build()` 生成文本。
  4. 使用 `deepseek_manager.client.chat.completions.create(...)` 获取大模型输出。
  5. 解析 JSON 数组，并通过 `st.expander` 等卡片组件，渲染出带颜色标签（Highly Recommended/Recommended 等）的药品卡片。

## Verification Plan

### Manual Verification
- 用户审批该计划后，我将直接生成该文件。
- 用户可在自己的终端，激活 conda 7606 环境，然后在 `app` 目录下运行 `streamlit run ui_module/streamlit_app.py` 来预览效果。如果需要修改页面颜色或微调布局，可以随时提出。
