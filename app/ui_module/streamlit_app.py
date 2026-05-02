import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

# Ensure other modules in the project can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_module import BERTManager
from evaluation_module import DrugRecommendationService
from static_module import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

STREAMLIT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "remote_llm_module"
    / "prompts"
    / "default_streamlit_followup.md"
)
FALLBACK_FOLLOW_UP_QUESTIONS = [
    "Have your symptoms been getting better, staying about the same, or getting worse?",
]
FALLBACK_ANSWER_OPTIONS = [
    "Getting better",
    "About the same",
    "Getting worse",
    "I'm not sure",
]

st.set_page_config(page_title="Smart Medical Assistant", page_icon="💊", layout="wide")

# --- Custom CSS Injection to beautify and hide Streamlit branding ---
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    if "conversation_rounds" not in st.session_state:
        st.session_state.conversation_rounds = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "diagnosis_input" not in st.session_state:
        st.session_state.diagnosis_input = ""


@st.cache_resource
def load_managers():
    """
    Use st.cache_resource to cache singleton managers,
    preventing repeated loading of massive models upon web page refresh or interaction.
    """
    bert = BERTManager(debug_mode=False)
    recommendation = DrugRecommendationService()
    return bert, recommendation


@st.cache_resource
def load_streamlit_followup_prompt() -> str:
    if STREAMLIT_PROMPT_PATH.exists():
        return STREAMLIT_PROMPT_PATH.read_text(encoding="utf-8")
    return ""


@st.cache_data
def load_system_metrics():
    """Load comparison metrics from the artifacts folder to display in the sidebar"""
    metrics_path = (
        Path.cwd()
        / "artifacts"
        / "exp_drug_recall"
        / "comparison_summary.json"
    )
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for comp in data.get("comparisons", []):
                    if (
                        comp.get("mode_b") == "candidate_union_no_prior_no_bm25"
                        or comp.get("mode_b") == "exp_candidate_union_no_prior_no_bm25"
                    ):
                        return {
                            "hit": comp.get(
                                "exp_candidate_union_no_prior_no_bm25_hit@20", 0.9843
                            ),
                            "recall": comp.get(
                                "exp_candidate_union_no_prior_no_bm25_recall@20", 0.9475
                            ),
                            "mrr": comp.get(
                                "exp_candidate_union_no_prior_no_bm25_mrr", 0.6313
                            ),
                        }
        except Exception:
            pass
    return {"hit": 0.9843, "recall": 0.9475, "mrr": 0.6313}


def queue_followup_query(question: str) -> None:
    st.session_state.pending_query = question


def reset_conversation() -> None:
    st.session_state.conversation_rounds = []
    st.session_state.pending_query = ""
    st.session_state.diagnosis_input = ""


def build_context_text(previous_rounds: list[dict[str, Any]], current_query: str) -> str:
    previous_user_questions = [
        str(round_data.get("user_text", "")).strip()
        for round_data in previous_rounds
        if str(round_data.get("user_text", "")).strip()
    ]
    if not previous_user_questions:
        return current_query

    history_lines = ["Previous health questions from this conversation:"]
    for idx, question in enumerate(previous_user_questions, start=1):
        history_lines.append(f"{idx}. {question}")
    history_lines.append(f"Current follow-up question: {current_query}")
    return "\n".join(history_lines)


def build_conversation_history(previous_rounds: list[dict[str, Any]]) -> str:
    if not previous_rounds:
        return "No previous conversation history."

    history_lines = []
    for idx, round_data in enumerate(previous_rounds, start=1):
        user_text = str(round_data.get("user_text", "")).strip()
        recommendations = round_data.get("recommendations", [])
        drug_names = [
            str(item.get("recommended_drug") or item.get("drug_name") or "").strip()
            for item in recommendations
            if str(item.get("recommended_drug") or item.get("drug_name") or "").strip()
        ]
        drug_summary = ", ".join(drug_names[:3]) if drug_names else "No drug recommendation available"
        history_lines.append(f"Round {idx} user question: {user_text}")
        history_lines.append(f"Round {idx} assistant summary: Recommended drugs included {drug_summary}.")
    return "\n".join(history_lines)


def find_latest_grounded_round(
    previous_rounds: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for round_data in reversed(previous_rounds):
        if str(round_data.get("round_type", "")).strip() == "grounded":
            return round_data
    return None


def extract_label_names(items: list[dict[str, Any]]) -> list[str]:
    names = []
    for item in items or []:
        name = str(item.get("name") or item.get("label") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def is_others_only_prediction(bert_prediction: dict[str, Any]) -> bool:
    disease_names = [
        name.lower() for name in extract_label_names(bert_prediction.get("diseases", []))
    ]
    return bool(disease_names) and all(name == "others" for name in disease_names)


def build_others_only_guidance(bert_prediction: dict[str, Any]) -> dict[str, Any]:
    symptom_names = extract_label_names(bert_prediction.get("symptoms", []))
    if symptom_names:
        summary = (
            "The current BERT result does not support a reliable disease-level match. "
            "Detected symptom clues include: "
            + ", ".join(symptom_names[:6])
            + "."
        )
    else:
        summary = (
            "The current BERT result does not support a reliable disease-level match, "
            "and no stable symptom labels were extracted from this round."
        )

    return {
        "symptom_names": symptom_names,
        "summary": summary,
        "advice": (
            "Because this round was classified as 'others', the system will not send the "
            "case to DeepSeek or provide medication recommendations. Please go to a "
            "hospital or clinic and seek help from a licensed medical professional."
        ),
    }


def normalize_follow_up_questions(questions: Any) -> list[str]:
    if isinstance(questions, str):
        questions = [questions]

    normalized = []
    for question in questions or []:
        text = str(question).strip()
        if text and text not in normalized:
            normalized.append(text)

    for fallback in FALLBACK_FOLLOW_UP_QUESTIONS:
        if len(normalized) >= 3:
            break
        if fallback not in normalized:
            normalized.append(fallback)
    return normalized[:3]


def normalize_answer_options(options: Any) -> list[str]:
    if isinstance(options, str):
        options = [options]

    normalized = []
    for option in options or []:
        text = str(option).strip()
        if text and text not in normalized:
            normalized.append(text)

    for fallback in FALLBACK_ANSWER_OPTIONS:
        if len(normalized) >= 4:
            break
        if fallback not in normalized:
            normalized.append(fallback)
    return normalized[:4]


def extract_json_string(raw_result: str) -> str:
    json_str = (raw_result or "").strip()
    if json_str.startswith("```"):
        lines = json_str.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        json_str = "\n".join(lines).strip()
    return json_str


def parse_deepseek_payload(
    raw_result: str,
) -> tuple[list[dict[str, Any]], str, str, list[str]]:
    json_str = extract_json_string(raw_result)
    payload = json.loads(json_str)

    if isinstance(payload, list):
        return (
            payload,
            "",
            FALLBACK_FOLLOW_UP_QUESTIONS[0],
            normalize_answer_options([]),
        )

    if not isinstance(payload, dict):
        raise ValueError("Unsupported DeepSeek payload type.")

    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("DeepSeek payload is missing recommendations.")

    assistant_reply = str(payload.get("assistant_reply", "")).strip()
    clarifying_question = str(
        payload.get("next_clarifying_question")
        or payload.get("clarifying_question")
        or payload.get("suggested_follow_up_question")
        or ""
    ).strip()
    if not clarifying_question:
        follow_up_questions = normalize_follow_up_questions(
            payload.get("suggested_follow_up_questions", [])
        )
        clarifying_question = (
            follow_up_questions[0] if follow_up_questions else FALLBACK_FOLLOW_UP_QUESTIONS[0]
        )

    answer_options = normalize_answer_options(
        payload.get("suggested_answer_options", [])
    )
    return recommendations, assistant_reply, clarifying_question, answer_options


def serialize_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def render_prompt_template(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", serialize_prompt_value(value))
    return rendered


def render_bert_analysis(bert_prediction: dict[str, Any]) -> None:
    st.subheader("🔬 AI Symptom Analysis Results")
    diseases = bert_prediction.get("diseases", [])
    symptoms = bert_prediction.get("symptoms", [])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔍 Predicted Diseases**")
        if diseases:
            disease_rows = [
                {
                    "Disease Name": d.get("name") or d.get("label", "Unknown"),
                    "Confidence": float(d.get("confidence", 0.0)),
                }
                for d in diseases
                if str(d.get("name") or d.get("label", "")).lower() != "others"
            ]
            if disease_rows:
                df_disease = pd.DataFrame(disease_rows)
                st.dataframe(
                    df_disease,
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
                    width="stretch",
                )
            else:
                st.info("No specific major diseases detected, identified as common discomfort.")
        else:
            st.info("No specific disease detected.")

    with col2:
        st.markdown("**🤒 Extracted Symptoms**")
        if symptoms:
            symptom_rows = [
                {
                    "Symptom Name": s.get("name") or s.get("label", "Unknown"),
                    "Confidence": float(s.get("confidence", 0.0)),
                }
                for s in symptoms
            ]
            if symptom_rows:
                df_symptom = pd.DataFrame(symptom_rows)
                st.dataframe(
                    df_symptom,
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
                    width="stretch",
                )
            else:
                st.info("No specific symptoms detected.")
        else:
            st.info("No specific symptoms detected.")
    st.divider()


def render_recommendation_cards(
    recommendations: list[dict[str, Any]],
    pipeline_output: dict[str, Any],
) -> None:
    st.subheader("📋 Recommended Medication Plan")

    if not recommendations:
        st.warning("No medication plan was generated for this round.")
        return

    pipeline_recs = pipeline_output.get("recommendations", [])
    pipeline_lookup = {
        str(item.get("drug_name", "")).lower(): item for item in pipeline_recs
    }

    for item in recommendations:
        drug_name = str(item.get("recommended_drug", "Unknown Drug"))
        preference = str(item.get("drug_preference", "unknown"))
        reason = str(item.get("recommendation_reasoning", "No description available"))

        internal_data = pipeline_lookup.get(drug_name.lower(), {})
        phase2_score = internal_data.get("phase2_score", None)
        matched_symptoms = internal_data.get("matched_symptoms", [])

        preference_lower = preference.lower()
        if "highly" in preference_lower or "强烈" in preference_lower:
            badge = "🟢 Highly Recommended"
        elif "optional" in preference_lower or "备选" in preference_lower:
            badge = "⚪ Optional"
        else:
            badge = "🔵 Recommended"

        with st.expander(f"{badge} : {drug_name}", expanded=True):
            if phase2_score is not None:
                cols = st.columns([1, 3])
                with cols[0]:
                    st.metric(label="Weight", value=f"{phase2_score:.2f}")
                with cols[1]:
                    if matched_symptoms:
                        st.markdown("**Targeted Symptoms:**")
                        tags = " ".join([f"`{symptom}`" for symptom in matched_symptoms])
                        st.markdown(tags)
                    else:
                        st.markdown("**Targeted Symptoms:**\n`General/Disease Match`")
                st.markdown("---")

            st.markdown(f"**DeepSeek Reasoning:**\n\n{reason}")


def render_others_only_guidance(guidance: dict[str, Any]) -> None:
    st.subheader("⚠️ Professional Medical Follow-up Recommended")
    st.warning(
        "BERT classified this round as `others`, so the app skipped both the medication "
        "ranking pipeline and the DeepSeek analysis."
    )
    st.markdown(guidance.get("summary", ""))

    symptom_names = guidance.get("symptom_names", [])
    if symptom_names:
        st.markdown("**Possible symptom clues detected in this round:**")
        for symptom_name in symptom_names:
            st.markdown(f"- `{symptom_name}`")
    else:
        st.info("No stable symptom clues were extracted for this round.")

    st.error(guidance.get("advice", "Please seek professional medical help promptly."))


def render_follow_up_section(
    assistant_reply: str,
    clarifying_question: str,
    answer_options: list[str],
    round_index: int,
) -> None:
    st.subheader("Continue The Conversation")
    if assistant_reply:
        st.markdown("**Assistant message**")
        st.info(assistant_reply)

    st.markdown("**Clarifying question**")
    st.write(clarifying_question or FALLBACK_FOLLOW_UP_QUESTIONS[0])
    st.caption("Choose one suggested answer below, or type your own reply in the input box above.")

    options = answer_options or normalize_answer_options([])
    columns = st.columns(len(options))
    for idx, question in enumerate(options):
        with columns[idx]:
            st.button(
                question,
                key=f"answer_option_round_{round_index}_{idx}",
                width="stretch",
                on_click=queue_followup_query,
                args=(question,),
            )


def render_recent_conversation(previous_rounds: list[dict[str, Any]]) -> None:
    if not previous_rounds:
        return

    st.subheader("Recent Conversation")
    st.caption("Brief history of previous rounds in this session.")

    for idx, round_data in enumerate(previous_rounds, start=1):
        round_type = str(round_data.get("round_type", "")).strip()
        user_text = str(round_data.get("user_text", "")).strip()
        bert_prediction = round_data.get("bert_prediction", {})
        disease_names = [
            str(item.get("name") or item.get("label") or "").strip()
            for item in bert_prediction.get("diseases", [])
            if str(item.get("name") or item.get("label") or "").strip()
            and str(item.get("name") or item.get("label") or "").strip().lower() != "others"
        ]
        recommendations = round_data.get("recommendations", [])
        drug_names = [
            str(item.get("recommended_drug") or item.get("drug_name") or "").strip()
            for item in recommendations
            if str(item.get("recommended_drug") or item.get("drug_name") or "").strip()
        ]

        label = f"Round {idx}: {user_text[:60]}{'...' if len(user_text) > 60 else ''}"
        with st.expander(label, expanded=False):
            st.markdown(f"**User Question:** {user_text}")
            if round_type == "followup_only":
                st.markdown(
                    "**Context Mode:** Follow-up only. This round reused previous conversation context and did not rerun BERT."
                )
            elif disease_names:
                st.markdown(f"**Detected Diseases:** {', '.join(disease_names[:3])}")
            else:
                st.markdown("**Detected Diseases:** No specific disease detected")

            if drug_names:
                st.markdown(f"**Recommended Drugs:** {', '.join(drug_names[:3])}")
            else:
                st.markdown("**Recommended Drugs:** No final drug recommendation")


def process_round(
    user_text: str,
    previous_rounds: list[dict[str, Any]],
    prompt_template: str,
    bert_manager: BERTManager,
    recommendation_manager: DrugRecommendationService,
    deepseek_client: OpenAI,
    system_prompt: str,
) -> dict[str, Any]:
    bert_placeholder = st.container()
    context_text = build_context_text(previous_rounds, user_text)

    with st.status("🩺 Processing your diagnosis...", expanded=True) as status:
        st.write("🔍 Analyzing symptoms and inferring diseases using BERT...")
        bert_prediction = bert_manager.predict(context_text)
        bert_payload = dict(bert_prediction)
        bert_payload["sentence"] = context_text

        with bert_placeholder:
            render_bert_analysis(bert_prediction)

        if is_others_only_prediction(bert_prediction):
            st.write(
                "⚠️ BERT classified this case as `others`. Skipping medication ranking and DeepSeek analysis."
            )
            status.update(
                label="Diagnosis complete! Professional follow-up recommended.",
                state="complete",
                expanded=False,
            )
            return {
                "round_type": "others_only",
                "user_text": user_text,
                "context_text": context_text,
                "bert_prediction": bert_payload,
                "pipeline_output": {"recommendations": []},
                "recommendations": [],
                "assistant_reply": "",
                "clarifying_question": "",
                "suggested_answer_options": [],
                "raw_result": "",
                "parse_error_message": "",
                "used_fallback_questions": False,
                "skip_deepseek_reason": "others_only",
                "hospital_guidance": build_others_only_guidance(bert_prediction),
            }

        st.write("💊 Retrieving and ranking candidate drugs...")
        pipeline_output = recommendation_manager.predict(bert_payload, flat_out=True)

        st.write("🤖 Requesting DeepSeek for diagnostic analysis...")
        prompt_content = render_prompt_template(
            prompt_template,
            current_user_input=user_text,
            context_text=context_text,
            conversation_history=build_conversation_history(previous_rounds),
            bert_output=bert_payload,
            pipeline_output=pipeline_output,
        )

        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content},
            ],
        )
        raw_result = response.choices[0].message.content or ""
        status.update(label="Diagnosis complete!", state="complete", expanded=False)

    parse_error_message = ""
    used_fallback_questions = False
    try:
        (
            recommendations,
            assistant_reply,
            clarifying_question,
            answer_options,
        ) = parse_deepseek_payload(raw_result)
    except (json.JSONDecodeError, ValueError) as exc:
        recommendations = []
        assistant_reply = ""
        clarifying_question = FALLBACK_FOLLOW_UP_QUESTIONS[0]
        answer_options = normalize_answer_options([])
        parse_error_message = str(exc)
        used_fallback_questions = True

    return {
        "round_type": "grounded",
        "user_text": user_text,
        "context_text": context_text,
        "bert_prediction": bert_payload,
        "pipeline_output": pipeline_output,
        "recommendations": recommendations,
        "assistant_reply": assistant_reply,
        "clarifying_question": clarifying_question,
        "suggested_answer_options": answer_options,
        "raw_result": raw_result,
        "parse_error_message": parse_error_message,
        "used_fallback_questions": used_fallback_questions,
        "skip_deepseek_reason": "",
        "hospital_guidance": {},
    }


def process_followup_round(
    user_text: str,
    previous_rounds: list[dict[str, Any]],
    prompt_template: str,
    deepseek_client: OpenAI,
    system_prompt: str,
) -> dict[str, Any]:
    grounded_round = find_latest_grounded_round(previous_rounds)
    reference_bert_output = grounded_round.get("bert_prediction", {}) if grounded_round else {}
    reference_pipeline_output = (
        grounded_round.get("pipeline_output", {}) if grounded_round else {"recommendations": []}
    )
    conversation_history = build_conversation_history(previous_rounds)
    context_text = (
        "Follow-up-only mode. Do not assume a new BERT prediction or a new drug-retrieval run "
        "for this round. Use the conversation history as the primary context. If reference "
        "BERT labels or reference candidate drugs are provided below, treat them as cached "
        "materials from the most recent grounded round, not as fresh predictions for the "
        "current sentence."
    )

    with st.status("💬 Processing follow-up question...", expanded=True) as status:
        st.write("🧾 Building conversation history context for the follow-up...")
        if grounded_round:
            st.write(
                "📌 Reusing the most recent grounded system context as reference material for DeepSeek..."
            )
        else:
            st.write("📌 No grounded round found. Sending conversation history only to DeepSeek...")

        st.write("🤖 Requesting DeepSeek for follow-up analysis...")
        prompt_content = render_prompt_template(
            prompt_template,
            current_user_input=(
                f"{user_text}\n\n"
                "Note: This is a follow-up turn. Do not assume the app reran BERT or the "
                "drug-retrieval pipeline for this message."
            ),
            context_text=context_text,
            conversation_history=conversation_history,
            bert_output=reference_bert_output,
            pipeline_output=reference_pipeline_output,
        )

        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content},
            ],
        )
        raw_result = response.choices[0].message.content or ""
        status.update(label="Follow-up complete!", state="complete", expanded=False)

    parse_error_message = ""
    used_fallback_questions = False
    try:
        (
            recommendations,
            assistant_reply,
            clarifying_question,
            answer_options,
        ) = parse_deepseek_payload(raw_result)
    except (json.JSONDecodeError, ValueError) as exc:
        recommendations = []
        assistant_reply = ""
        clarifying_question = FALLBACK_FOLLOW_UP_QUESTIONS[0]
        answer_options = normalize_answer_options([])
        parse_error_message = str(exc)
        used_fallback_questions = True

    return {
        "round_type": "followup_only",
        "user_text": user_text,
        "context_text": context_text,
        "bert_prediction": {},
        "pipeline_output": {"recommendations": []},
        "recommendations": recommendations,
        "assistant_reply": assistant_reply,
        "clarifying_question": clarifying_question,
        "suggested_answer_options": answer_options,
        "raw_result": raw_result,
        "parse_error_message": parse_error_message,
        "used_fallback_questions": used_fallback_questions,
        "skip_deepseek_reason": "",
        "hospital_guidance": {},
        "grounding_reference_available": bool(grounded_round),
        "grounding_bert_prediction": reference_bert_output,
        "grounding_pipeline_output": reference_pipeline_output,
    }


init_session_state()
bert_manager, recommendation_manager = load_managers()
streamlit_followup_prompt = load_streamlit_followup_prompt()
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
    # st.markdown("### 📊 Offline Evaluation Metrics")
    # st.metric(label="Hit@20 (Accuracy)", value=f"{metrics['hit']*100:.2f}%")
    # st.metric(label="Recall@20", value=f"{metrics['recall']*100:.2f}%")
    # st.metric(label="MRR (Mean Reciprocal Rank)", value=f"{metrics['mrr']:.4f}")

    st.markdown("---")
    st.caption("Powered by BERT Multi-task Classifier & XGBoost/Deterministic Ranker Pipeline.")

# --- Main Page ---
st.title("💊 Smart Medical Assistant")
st.markdown(
    "Please describe your symptoms. The system will automatically analyze them, match relevant medications, and provide professional medical guidance."
)
if st.session_state.conversation_rounds:
    st.button("Start New Conversation", on_click=reset_conversation)

active_query = ""
triggered_by_followup = False
pending_query = str(st.session_state.pending_query or "").strip()
if pending_query:
    active_query = pending_query
    triggered_by_followup = True
    st.session_state.pending_query = ""

with st.form(key="diagnosis_form"):
    st.text_area(
        (
            "✍️ Reply in your own words:"
            if st.session_state.conversation_rounds
            else "✍️ Please describe your symptoms in detail:"
        ),
        height=150,
        key="diagnosis_input",
        placeholder=(
            "Type your own answer to continue the conversation..."
            if st.session_state.conversation_rounds
            else "Example: I've had a headache and fever for the past few days, with a runny nose..."
        ),
    )
    submit_button = st.form_submit_button(
        "💬 Continue Conversation" if st.session_state.conversation_rounds else "🚀 Start Diagnosis",
        width="stretch",
    )

if submit_button:
    submitted_query = str(st.session_state.diagnosis_input).strip()
    if not submitted_query:
        st.warning("Please enter your symptom description!")
    else:
        active_query = submitted_query
        triggered_by_followup = bool(st.session_state.conversation_rounds)

processed_round_this_run = False
if active_query:
    if not streamlit_followup_prompt:
        st.error(
            f"Failed to construct Streamlit prompt. Missing file: {STREAMLIT_PROMPT_PATH}"
        )
    else:
        try:
            if triggered_by_followup:
                current_round = process_followup_round(
                    user_text=active_query,
                    previous_rounds=st.session_state.conversation_rounds,
                    prompt_template=streamlit_followup_prompt,
                    deepseek_client=deepseek_client,
                    system_prompt=system_prompt,
                )
            else:
                current_round = process_round(
                    user_text=active_query,
                    previous_rounds=st.session_state.conversation_rounds,
                    prompt_template=streamlit_followup_prompt,
                    bert_manager=bert_manager,
                    recommendation_manager=recommendation_manager,
                    deepseek_client=deepseek_client,
                    system_prompt=system_prompt,
                )
        except Exception as exc:
            st.error(f"Failed to call DeepSeek API: {exc}")
        else:
            st.session_state.conversation_rounds.append(current_round)
            processed_round_this_run = True

conversation_rounds = st.session_state.conversation_rounds
current_round = conversation_rounds[-1] if conversation_rounds else None
previous_rounds = conversation_rounds[:-1] if len(conversation_rounds) > 1 else []

if current_round:
    if (
        not processed_round_this_run
        and current_round.get("round_type") in {"grounded", "others_only"}
    ):
        render_bert_analysis(current_round.get("bert_prediction", {}))

    if current_round.get("round_type") == "followup_only":
        st.info(
            "This follow-up turn used conversation history only. The app did not rerun BERT or the medication retrieval pipeline for this round."
        )

    if current_round.get("skip_deepseek_reason") == "others_only":
        render_others_only_guidance(current_round.get("hospital_guidance", {}))
    else:
        render_recommendation_cards(
            current_round.get("recommendations", []),
            current_round.get("pipeline_output", {}),
        )

        if current_round.get("parse_error_message"):
            st.warning(
                "DeepSeek returned a non-standard JSON format for this round. Generic English conversation prompts are shown below."
            )
            with st.expander("View raw response text"):
                st.code(current_round.get("raw_result", ""))

        render_follow_up_section(
            current_round.get("assistant_reply", ""),
            current_round.get("clarifying_question", ""),
            current_round.get("suggested_answer_options", []),
            round_index=len(conversation_rounds),
        )
    render_recent_conversation(previous_rounds)
