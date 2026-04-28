# 系统/第三方模块导入
import os
from typing import Optional, Any
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 本地模块导入
from static_module import TaskStatus, PROJECT_NAME, THREAD_TIMEOUT
from utility_module import logger
from remote_llm_module import DeepSeekManager
from deployment_module import BERTManager
from evaluation_module import DrugRecommendationService

# 初始化 FastAPI 应用
app = FastAPI(title="医学分析与推荐系统")

# 允许跨域请求（如果前端代码和后端不在同一个端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例管理器（在 Web 服务器启动时加载，保留在内存中）
deepseek_manager = DeepSeekManager(debug_mode=False)
bert_manager = BERTManager(debug_mode=False)
recommendation_manager = DrugRecommendationService()

# 定义前端发送过来的数据格式
class UserQuery(BaseModel):
    message: str

@app.post("/api/analyze")
async def analyze_input(query: UserQuery):
    """
    接收前端用户输入，经过 BERT 和推荐系统处理后，调用 DeepSeek 获取最终结果
    """
    user_input = query.message.strip()
    if not user_input:
        return {"status": "error", "message": "输入不能为空"}

    try:
        # 1. BERT 预测
        bert_prediction = bert_manager.predict(user_input)
        logger.debug(f"bert_prediction: {bert_prediction}")

        # 2. 推荐系统处理
        pipeline_output = recommendation_manager.predict(bert_prediction, flat_out=True)
        logger.debug(f"pipeline_output: {pipeline_output}")

        # 3. 组装数据并向 DeepSeek 请求
        # 【注意】这里的逻辑需要根据你实际的 DeepSeekManager 进行调整。
        # 原代码中使用的是 .send() 发送异步任务，但在标准 HTTP 请求中，
        # 我们通常需要等待 (await) DeepSeek 生成完毕后，直接获取返回值。
        # 假设你的管理器提供了一个类似 generate_response 的同步或异步方法：

        deepseek_response = await deepseek_manager.generate_response({
            "sentences": user_input,
            "pipeline_output": pipeline_output,
            "bert_output": bert_prediction,
        })

        # 4. 返回 JSON 数据给前端
        return {
            "status": "success",
            "user_input": user_input,
            "bert_output": bert_prediction,
            "pipeline_output": pipeline_output,
            "deepseek_response": deepseek_response
        }

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return {"status": "error", "message": str(e)}

# 为了快速测试，提供一个极其简单的内嵌 HTML 前端页面
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>智能诊断与推荐</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            .chat-box { border: 1px solid #ccc; padding: 20px; min-height: 300px; margin-bottom: 20px; background: #f9f9f9; }
            .debug-info { font-size: 0.85em; color: #666; margin-top: 5px; }
            textarea { width: 100%; height: 80px; padding: 10px; margin-bottom: 10px; }
            button { padding: 10px 20px; cursor: pointer; background: #007bff; color: white; border: none; }
            button:disabled { background: #ccc; }
        </style>
    </head>
    <body>
        <h2>智能诊断与推荐系统</h2>
        <div class="chat-box" id="chatBox"></div>
        <textarea id="userInput" placeholder="请输入患者症状或文本描述..."></textarea>
        <button id="sendBtn" onclick="sendMessage()">发送</button>

        <script>
            async function sendMessage() {
                const inputEl = document.getElementById('userInput');
                const btn = document.getElementById('sendBtn');
                const chatBox = document.getElementById('chatBox');
                const text = inputEl.value.trim();
                
                if (!text) return;

                // 在界面上显示用户输入
                chatBox.innerHTML += `<p><strong>用户:</strong> ${text}</p>`;
                inputEl.value = '';
                btn.disabled = true;
                btn.innerText = '处理中...';

                try {
                    // 发送 POST 请求到后端的 FastAPI 路由
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        chatBox.innerHTML += `
                            <p><strong>DeepSeek回复:</strong> ${data.deepseek_response}</p>
                            <div class="debug-info">
                                [内部数据] BERT输出: ${JSON.stringify(data.bert_output)} <br>
                                [内部数据] 推荐输出: ${JSON.stringify(data.pipeline_output)}
                            </div>
                            <hr>
                        `;
                    } else {
                        chatBox.innerHTML += `<p style="color:red;"><strong>错误:</strong> ${data.message}</p>`;
                    }
                } catch (err) {
                    chatBox.innerHTML += `<p style="color:red;"><strong>网络错误:</strong> 请检查后端是否运行</p>`;
                } finally {
                    btn.disabled = false;
                    btn.innerText = '发送';
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            }
        </script>
    </body>
    </html>
    """
def app_run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    启动 Web 服务的入口函数
    """
    logger.info(f"正在启动 {PROJECT_NAME} Web 后端...")
    # 这里直接调用 uvicorn 启动 FastAPI 实例
    uvicorn.run(app, host=host, port=port)
if __name__ == "__main__":
    # 启动服务器，默认运行在 http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)