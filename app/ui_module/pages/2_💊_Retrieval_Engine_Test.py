import os
import sys
import streamlit as st
import pandas as pd
import json

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation_module import DrugRecommendationService

st.set_page_config(page_title="Retrieval Engine Tester", page_icon="💊", layout="wide")

@st.cache_resource
def load_recommender():
    return DrugRecommendationService()

st.title("💊 Multi-stage Drug Retrieval Engine Test")
st.markdown("This debug page tests the backend ranking pipeline (`Phase2FinalRecommender`). It bypasses BERT and DeepSeek, allowing you to manually inject simulated disease and symptom tags to observe how the ranking system scores candidates.")

recommender = load_recommender()

with st.sidebar:
    st.header("Pipeline Mode")
    st.info(f"Current Engine Mode:\n\n**`{recommender.phase2_mode}`**")
    
with st.form("retrieval_test_form"):
    st.markdown("### 💉 Inject Semantic Tags")
    col1, col2 = st.columns(2)
    with col1:
        disease_input = st.text_input("Diseases (comma-separated)", value="common_cold, cough")
    with col2:
        symptom_input = st.text_input("Symptoms (comma-separated)", value="fever, runny_nose")
        
    st.markdown("### 🛠️ Inject Raw BERT Output JSON (Optional Override)")
    raw_json_input = st.text_area("If provided, this JSON overrides the fields above.", value="", height=100, placeholder='{"diseases": [{"name": "headache", "confidence": 0.9}], "symptoms": []}')
    
    submitted = st.form_submit_button("Run Retrieval & Ranking")

if submitted:
    bert_simulated = None
    
    if raw_json_input.strip():
        try:
            bert_simulated = json.loads(raw_json_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON format in override field.")
            st.stop()
    else:
        # Construct from comma-separated fields
        d_list = [{"name": d.strip(), "confidence": 1.0} for d in disease_input.split(",") if d.strip()]
        s_list = [{"name": s.strip(), "confidence": 1.0} for s in symptom_input.split(",") if s.strip()]
        bert_simulated = {
            "diseases": d_list,
            "symptoms": s_list,
            "need_first_aid": 0
        }
        
    st.subheader("📥 Constructed Input Payload")
    st.json(bert_simulated)
    
    with st.spinner("Executing retrieval pipeline..."):
        # We want the flat output to easily display in a table
        pipeline_output = recommender.predict(bert_simulated, flat_out=True)
        
    st.subheader("📤 Ranked Recommendations")
    recs = pipeline_output.get("recommendations", [])
    
    if recs:
        df = pd.DataFrame(recs)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No recommendations retrieved. The disease might be filtered (e.g. 'others') or missing from candidate set.")
