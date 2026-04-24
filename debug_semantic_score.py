"""Diagnose why semantic_score is 0 in final results."""
import sys, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

from dotenv import load_dotenv
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "dummy"

from fastapi_module.service import get_recommendation_service

service = get_recommendation_service()
service.ensure_ready()
pipeline = service.pipeline

symptom_text = "I have big, painful bumps on my skin that leave scars. How can I treat this?"
diseases = [{"name": "acne", "confidence": 1.0}]
symptoms = [{"name": "nodal_skin_eruptions", "confidence": 1.0}, {"name": "scurring", "confidence": 1.0}]

# Step 1: Semantic recall
sem_cands = pipeline.semantic_recall(symptom_text, top_k=300)
print("=== SEMANTIC RECALL ===")
print(f"Count: {len(sem_cands)}")
print(f"semantic_score range: [{sem_cands['semantic_score'].min():.4f}, {sem_cands['semantic_score'].max():.4f}]")
print(f"Top-5 drugs:")
print(sem_cands[['drug_name', 'semantic_score', 'semantic_score_raw']].head(5).to_string())
# Check if ground truth drugs are in semantic recall
gt_drugs = ["doxycycline", "clindamycin", "sulfamethoxazole / trimethoprim"]
for d in gt_drugs:
    match = sem_cands[sem_cands['drug_name'] == d]
    if len(match) > 0:
        print(f"  GT drug '{d}' in semantic top-300: score={match['semantic_score'].values[0]:.4f}")
    else:
        print(f"  GT drug '{d}' NOT in semantic top-300")

# Step 2: Label recall
lab_cands = pipeline.label_recall(diseases, symptoms, top_k=300)
print(f"\n=== LABEL RECALL ===")
print(f"Count: {len(lab_cands)}")
print(f"label_score range: [{lab_cands['label_score'].min():.4f}, {lab_cands['label_score'].max():.4f}]")
print(f"Top-5 drugs:")
print(lab_cands[['drug_name', 'label_score', 'disease_conf_overlap', 'symptom_conf_overlap']].head(5).to_string())
for d in gt_drugs:
    match = lab_cands[lab_cands['drug_name'] == d]
    if len(match) > 0:
        print(f"  GT drug '{d}' in label top-300: score={match['label_score'].values[0]:.4f}")
    else:
        print(f"  GT drug '{d}' NOT in label top-300")

# Step 3: Check overlap
sem_idx = set(sem_cands.index)
lab_idx = set(lab_cands.index)
overlap = sem_idx & lab_idx
print(f"\n=== OVERLAP ANALYSIS ===")
print(f"Semantic indices: {len(sem_idx)}")
print(f"Label indices: {len(lab_idx)}")
print(f"Overlap: {len(overlap)} drugs appear in BOTH channels")
print(f"Only semantic: {len(sem_idx - lab_idx)}")
print(f"Only label: {len(lab_idx - sem_idx)}")

# Step 4: Fuse
fused = pipeline.fuse_recalls(sem_cands, lab_cands, semantic_weight=0.5, label_weight=0.5, top_k=300)
print(f"\n=== FUSED (Top-300) ===")
print(f"Count: {len(fused)}")
has_sem = (fused['semantic_score'] > 0).sum()
has_lab = (fused['label_score'] > 0).sum()
has_both = ((fused['semantic_score'] > 0) & (fused['label_score'] > 0)).sum()
print(f"With semantic_score > 0: {has_sem}")
print(f"With label_score > 0: {has_lab}")
print(f"With BOTH > 0: {has_both}")
print(f"\nFused top-10 (pre-rerank):")
print(fused[['drug_name', 'recall_fused_score', 'semantic_score', 'label_score']].head(10).to_string())
print(f"\nFused bottom-10:")
print(fused[['drug_name', 'recall_fused_score', 'semantic_score', 'label_score']].tail(10).to_string())

# Check if any fused drug has both scores
both_mask = (fused['semantic_score'] > 0) & (fused['label_score'] > 0)
if both_mask.sum() > 0:
    print(f"\nDrugs with BOTH scores:")
    print(fused[both_mask][['drug_name', 'recall_fused_score', 'semantic_score', 'label_score']].head(10).to_string())
