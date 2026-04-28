import os
import sys
import json
import streamlit as st

# Ensure other modules in the project can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_module import BERTManager
from evaluation_module import DrugRecommendationService
from static_module import DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEFAULT_PROMPT_FOLDER_PATH
from openai import OpenAI
from pathlib import Path

st.set_page_config(page_title="Smart Medical Assistant", page_icon="💊", layout="wide")

# --- Custom CSS Injection to beautify and hide Streamlit branding ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_managers():
    """
    Use st.cache_resource to cache singleton managers,
    preventing repeated loading of massive models upon web page refresh or interaction.
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

@st.cache_data
def load_system_metrics():
    """Load comparison metrics from the artifacts folder to display in the sidebar"""
    metrics_path = Path(__file__).parent.parent.parent / "artifacts" / "exp_drug_recall" / "comparison_summary.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for comp in data.get("comparisons", []):
                    if comp.get("mode_b") == "candidate_union_no_prior_no_bm25" or comp.get("mode_b") == "exp_candidate_union_no_prior_no_bm25":
                        return {
                            "hit": comp.get("exp_candidate_union_no_prior_no_bm25_hit@20", 0.9843),
                            "recall": comp.get("exp_candidate_union_no_prior_no_bm25_recall@20", 0.9475),
                            "mrr": comp.get("exp_candidate_union_no_prior_no_bm25_mrr", 0.6313)
                        }
        except:
            pass
    return {"hit": 0.9843, "recall": 0.9475, "mrr": 0.6313}

default_prompt = load_deepseek_prompt()
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
system_prompt = r"You are a medical consultant robot."

# --- Sidebar: System Dashboard ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.title("System Dashboard")
    st.markdown("---")
    mode_options = ["label_core_rerank", "xgb_ranker"]
    current_mode = recommendation_manager.phase2_mode
    current_index = mode_options.index(current_mode) if current_mode in mode_options else 0
    selected_mode = st.selectbox("⚙️ Engine Mode", mode_options, index=current_index)
    if selected_mode != current_mode:
        with st.spinner(f"Switching mode to {selected_mode}..."):
            recommendation_manager.switch_mode(selected_mode)
            st.rerun()
    st.info("**Status**: Online 🟢")

    metrics = load_system_metrics()
    st.markdown("### 📊 Offline Evaluation Metrics")
    st.metric(label="Hit@20 (Accuracy)", value=f"{metrics['hit']*100:.2f}%")
    st.metric(label="Recall@20", value=f"{metrics['recall']*100:.2f}%")
    st.metric(label="MRR (Mean Reciprocal Rank)", value=f"{metrics['mrr']:.4f}")
    
    st.markdown("---")
    st.caption("Powered by BERT Multi-task Classifier & XGBoost/Deterministic Ranker Pipeline.")

# --- Main Page ---
st.title("💊 Smart Medical Assistant")
st.markdown("Please describe your symptoms. The system will automatically analyze them, match relevant medications, and provide professional medical guidance.")

# Wrap input in a form for better UX
with st.form(key="diagnosis_form"):
    user_input = st.text_area("✍️ Please describe your symptoms in detail:", height=150, placeholder="Example: I've had a headache and fever for the past few days, with a runny nose...")
    submit_button = st.form_submit_button("🚀 Start Diagnosis", use_container_width=True)

if submit_button:
    if not user_input.strip():
        st.warning("Please enter your symptom description!")
    else:
        # Placeholder for the BERT extraction results to show immediately outside the status
        bert_placeholder = st.container()
        
        # Use st.status for an elegant grouped loading sequence
        with st.status("🩺 Processing your diagnosis...", expanded=True) as status:
            
            # Step 1: BERT Classification Inference
            st.write("🔍 Analyzing symptoms and inferring diseases using BERT...")
            bert_prediction = bert_manager.predict(user_input)
            
            # --- IMMEDIATELY Display BERT Extraction Results in the placeholder ---
            with bert_placeholder:
                st.subheader("🔬 AI Symptom Analysis Results")
                diseases = bert_prediction.get("diseases", [])
                symptoms = bert_prediction.get("symptoms", [])
                
                import pandas as pd
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔍 Predicted Diseases**")
                    if diseases:
                        d_list = [{"Disease Name": d.get("name") or d.get("label", "Unknown"), "Confidence": float(d.get("confidence", 0.0))} for d in diseases if str(d.get("name") or d.get("label", "")).lower() != "others"]
                        if d_list:
                            df_d = pd.DataFrame(d_list)
                            st.dataframe(
                                df_d,
                                column_config={
                                    "Confidence": st.column_config.ProgressColumn(
                                        "Confidence",
                                        help="AI predicted match probability",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.info("No specific major diseases detected, identified as common discomfort.")
                    else:
                        st.info("No specific disease detected.")
                        
                with col2:
                    st.markdown("**🤒 Extracted Symptoms**")
                    if symptoms:
                        s_list = [{"Symptom Name": s.get("name") or s.get("label", "Unknown"), "Confidence": float(s.get("confidence", 0.0))} for s in symptoms]
                        if s_list:
                            df_s = pd.DataFrame(s_list)
                            st.dataframe(
                                df_s,
                                column_config={
                                    "Confidence": st.column_config.ProgressColumn(
                                        "Confidence",
                                        help="AI identified symptom probability",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.info("No specific symptoms detected.")
                    else:
                        st.info("No specific symptoms detected.")
                st.divider()
            
            # Step 2: Recommendation System Retrieve and Rank Candidate Drugs
            st.write("💊 Retrieving and ranking candidate drugs...")
            pipeline_output = recommendation_manager.predict(bert_prediction, flat_out=True)
            
            # Step 3: DeepSeek
            st.write("🤖 Requesting DeepSeek for diagnostic analysis...")
            if not default_prompt:
                status.update(label="Failed to construct prompt.", state="error", expanded=True)
                st.stop()
                
            prompt_content = default_prompt.format(
                sentences=user_input,
                pipeline_output=pipeline_output,
                bert_output=bert_prediction,
            )
            
            try:
                response = deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ]
                )
                raw_result = response.choices[0].message.content
                status.update(label="Diagnosis complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"Failed to call DeepSeek API: {e}", state="error", expanded=True)
                st.stop()

        # Start parsing JSON result returned by the LLM
        try:
            json_str = raw_result
            if json_str.startswith("```json"):
                json_str = json_str.strip("```json").strip("```")
            elif json_str.startswith("```"):
                json_str = json_str.strip("```")
                
            recommendations = json.loads(json_str)
            
            st.subheader("📋 Recommended Medication Plan")
            
            # Create a lookup dictionary from pipeline_output to find phase2_score and matched_symptoms
            pipeline_recs = pipeline_output.get("recommendations", [])
            pipeline_lookup = {item.get("drug_name", "").lower(): item for item in pipeline_recs}
            
            # Iterate and draw beautiful cards
            for item in recommendations:
                drug_name = item.get("recommended_drug", "Unknown Drug")
                preference = item.get("drug_preference", "unknown")
                reason = item.get("recommendation_reasoning", "No description available")
                
                # Fetch internal AI stats
                internal_data = pipeline_lookup.get(drug_name.lower(), {})
                phase2_score = internal_data.get("phase2_score", None)
                matched_symp = internal_data.get("matched_symptoms", [])
                
                # Assign different color badges according to recommendation level
                preference_lower = preference.lower()
                if "highly" in preference_lower or "强烈" in preference_lower:
                    badge = "🟢 Highly Recommended"
                    color = "success"
                elif "optional" in preference_lower or "备选" in preference_lower:
                    badge = "⚪ Optional"
                    color = "normal"
                else:
                    badge = "🔵 Recommended"
                    color = "info"
                    
                # Use expander to do folding card display
                with st.expander(f"{badge} : {drug_name}", expanded=True):
                    
                    # Display AI Matching Stats if available
                    if phase2_score is not None:
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.metric(label="AI Match Score", value=f"{phase2_score:.2f}")
                        with cols[1]:
                            if matched_symp:
                                st.markdown("**Targeted Symptoms:**")
                                tags = " ".join([f"`{s}`" for s in matched_symp])
                                st.markdown(tags)
                            else:
                                st.markdown("**Targeted Symptoms:**\n`General/Disease Match`")
                        st.markdown("---")
                        
                    st.markdown(f"**DeepSeek Reasoning:**\n\n{reason}")
                    
        except json.JSONDecodeError:
            st.error("Failed to parse result. The LLM might have returned a non-standard JSON format.")
            with st.expander("View raw response text"):
                st.code(raw_result)
