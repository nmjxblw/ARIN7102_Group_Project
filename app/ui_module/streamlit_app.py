import os
import sys
import json
import streamlit as st

# 确保能正确导入同一项目下的其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_module import BERTManager
from evaluation_module import DrugRecommendationService
from static_module import DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEFAULT_PROMPT_FOLDER_PATH
from openai import OpenAI
from pathlib import Path

st.set_page_config(page_title="智能医药助手", page_icon="💊", layout="centered")

@st.cache_resource
def load_managers():
    """
    使用 st.cache_resource 缓存单例管理器，
    防止在每次网页刷新或交互时重复加载庞大的模型。
    """
    bert = BERTManager(debug_mode=False)
    recommendation = DrugRecommendationService()
    return bert, recommendation

bert_manager, recommendation_manager = load_managers()

@st.cache_resource
def load_deepseek_prompt():
    prompt_file = None
    for root, dir, files in os.walk(DEFAULT_PROMPT_FOLDER_PATH):
        for file in files:
            if file.endswith(".md") and file.startswith("default"):
                prompt_file = Path(root) / file
                break
    if prompt_file and prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

default_prompt = load_deepseek_prompt()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
system_prompt = r"You are a medical consultant robot."

st.title("💊 智能医药助手诊断系统")
st.markdown("请输入您的症状描述，系统将自动分析、为您匹配相关药物并提供专业的用药指导。")

user_input = st.text_area("✍️ 请详细描述您的症状：", height=150, placeholder="例如：我最近几天头疼发烧，还一直流清鼻涕...")

if st.button("🚀 开始诊断", use_container_width=True):
    if not user_input.strip():
        st.warning("请输入病情描述！")
    else:
        # Step 1: BERT 分类推断
        with st.spinner("🔍 正在使用 BERT 分析症状并推断疾病..."):
            bert_prediction = bert_manager.predict(user_input)
            
        # Step 2: 推荐系统检索并排序候选药物
        with st.spinner("💊 正在检索并排序候选药物..."):
            pipeline_output = recommendation_manager.predict(bert_prediction, flat_out=True)
            
        # Step 3: DeepSeek 包装与解释展示
        with st.spinner("🤖 正在请求 DeepSeek 提供诊断解析..."):
            # 组装 Prompt
            if not default_prompt:
                st.error("无法构建有效的提示词模板。")
                st.stop()
                
            prompt_content = default_prompt.format(
                sentences=user_input,
                pipeline_output=pipeline_output,
                bert_output=bert_prediction,
            )
            
            # 发起同步请求 (阻塞等待结果)
            try:
                response = deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ]
                )
                raw_result = response.choices[0].message.content
                
                # 开始解析大模型返回的 JSON 结果
                try:
                    # 容错处理：如果大模型带了 markdown 代码块前缀
                    json_str = raw_result
                    if json_str.startswith("```json"):
                        json_str = json_str.strip("```json").strip("```")
                    elif json_str.startswith("```"):
                        json_str = json_str.strip("```")
                        
                    recommendations = json.loads(json_str)
                    
                    st.divider()
                    st.subheader("📋 推荐用药方案")
                    
                    # 遍历并绘制漂亮的卡片
                    for item in recommendations:
                        drug_name = item.get("recommended_drug", "未知药物")
                        preference = item.get("drug_preference", "unknown")
                        reason = item.get("recommendation_reasoning", "暂无说明")
                        
                        # 根据推荐级别分配不同的颜色徽章
                        preference_lower = preference.lower()
                        if "highly" in preference_lower or "强烈" in preference_lower:
                            badge = "🟢 强烈推荐"
                        elif "optional" in preference_lower or "备选" in preference_lower:
                            badge = "⚪ 备选"
                        else:
                            badge = "🔵 推荐"
                            
                        # 使用 expander 做折叠卡片展示
                        with st.expander(f"{badge} : {drug_name}", expanded=True):
                            st.markdown(f"**推荐理由：**\n\n{reason}")
                            
                except json.JSONDecodeError:
                    st.error("解析结果失败，大模型可能返回了非标准 JSON 格式。")
                    with st.expander("查看原始返回文本"):
                        st.code(raw_result)
                        
            except Exception as e:
                st.error(f"调用 DeepSeek 接口失败: {e}")
