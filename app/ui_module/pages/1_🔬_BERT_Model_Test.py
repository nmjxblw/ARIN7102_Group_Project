import os
import sys
import streamlit as st
import pandas as pd

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deployment_module import BERTManager

st.set_page_config(page_title="BERT Model Tester", page_icon="🔬", layout="wide")

@st.cache_resource
def load_bert():
    return BERTManager(debug_mode=False)

st.title("🔬 BERT Multi-Task Classifier Test")
st.markdown("This debug page bypasses the full pipeline and only interacts with the **BERT model**. Use this to test natural language understanding, entity extraction, and intent classification.")

bert_manager = load_bert()

with st.form("bert_test_form"):
    user_text = st.text_area("Enter raw patient text:", "I've been coughing a lot and feeling hot.", height=100)
    submitted = st.form_submit_button("Run Inference")

if submitted:
    if user_text.strip():
        with st.spinner("Running BERT inference..."):
            prediction = bert_manager.predict(user_text)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Extracted Diseases")
            diseases = prediction.get("diseases", [])
            if diseases:
                df_d = pd.DataFrame(diseases)
                st.dataframe(df_d, use_container_width=True)
            else:
                st.info("No diseases extracted.")
                
            st.subheader("Extracted Symptoms")
            symptoms = prediction.get("symptoms", [])
            if symptoms:
                df_s = pd.DataFrame(symptoms)
                st.dataframe(df_s, use_container_width=True)
            else:
                st.info("No symptoms extracted.")
                
        with col2:
            st.subheader("Raw JSON Output")
            st.json(prediction)
    else:
        st.warning("Please enter some text.")
