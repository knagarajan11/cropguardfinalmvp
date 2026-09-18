from pathlib import Path
from PIL import Image

from vision.nemotron_omni import NemotronOmniConfig, NemotronOmniVisionModel
from agents.diagnosis_agent import DiagnosisAgent
from agents.context_agent import ContextAgent, FarmerProfile
from agents.knowledge_agent import KnowledgeAgent
from agents.generation_agent import GroundedGenerationAgent


BASE_MODEL = Path(
    "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2"
)

LORA_MODEL = Path(
    "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/model_remapped"
)

IMAGE_PATH = Path(
    "/workspace/data/sft/images/"
    "plantdoc_f7558468-ac52-452a-bc09-d68fa3d82ecb.jpg"
)


print("=" * 70)
print("CropGuard - REAL GROUNDED GENERATION TEST")
print("=" * 70)

print("\n[1] Paths")
print("Base :", BASE_MODEL)
print("LoRA :", LORA_MODEL)
print("Image:", IMAGE_PATH)

assert BASE_MODEL.is_dir(), f"Base model missing: {BASE_MODEL}"
assert LORA_MODEL.is_dir(), f"LoRA model missing: {LORA_MODEL}"
assert IMAGE_PATH.is_file(), f"Image missing: {IMAGE_PATH}"

# ------------------------------------------------------------------
# 2. Load frozen validated Nemotron Omni + CropGuard LoRA
# ------------------------------------------------------------------
print("\n[2] Loading frozen Nemotron Omni + CropGuard LoRA...")

config = NemotronOmniConfig(
    base_model_path=BASE_MODEL,
    lora_adapter_path=LORA_MODEL,
    device="cuda:0",
    dtype="bfloat16",
    max_new_tokens=1536,
    expected_lora_modules=116,
)

model = NemotronOmniVisionModel(config)
model.load()

print("MODEL LOAD: PASS")

# ------------------------------------------------------------------
# 3. Diagnosis Agent
# ------------------------------------------------------------------
print("\n[3] Running Diagnosis Agent...")

diagnosis_agent = DiagnosisAgent(model=model)
image = Image.open(IMAGE_PATH).convert("RGB")

diagnosis = diagnosis_agent.diagnose(
    image=image,
    farmer_query=(
        "Analyze the crop image and identify the crop or plant, "
        "the most likely disease, visible symptoms, and confidence. "
        "Do not provide treatment recommendations."
    ),
)

print("Diagnosis success :", diagnosis.success)
print("Crop             :", diagnosis.crop)
print("Disease          :", diagnosis.disease)
print("Symptoms         :", diagnosis.symptoms)
print("Confidence       :", diagnosis.confidence)

print("\n--- MODEL FINAL TEXT ---")
print(diagnosis.final_text)

print("\n--- MODEL RAW TEXT ---")
print(diagnosis.raw_text)

assert diagnosis.success, "Diagnosis failed"
assert diagnosis.crop, "Crop not detected"
assert diagnosis.disease, "Disease not detected"

# ------------------------------------------------------------------
# 4. Context Agent
# ------------------------------------------------------------------
print("\n[4] Building farmer/context state...")

context_agent = ContextAgent()

context_agent.set_farmer_profile(
    FarmerProfile(
        farmer_id="demo-farmer",
        name="Demo Farmer",
        preferred_language="English",
        location="Tamil Nadu",
    )
)

context_agent.set_recommendation_preference("General")

context_agent.start_new_case(
    image_path=str(IMAGE_PATH),
    farmer_query="What should I do for this crop disease?",
)

context_agent.update_diagnosis(
    crop=diagnosis.crop,
    disease=diagnosis.disease,
    symptoms=diagnosis.symptoms,
    confidence=diagnosis.confidence,
)

context = context_agent.build_context()

print("Recommendation preference:",
      context.recommendation_preference)
print("Context crop             :", context.current_case.crop)
print("Context disease          :", context.current_case.disease)

# ------------------------------------------------------------------
# 5. Knowledge Agent
# ------------------------------------------------------------------
print("\n[5] Running Hybrid Retrieval + NVIDIA Reranker + Evidence Filter...")

knowledge_agent = KnowledgeAgent(
    min_reranker_score=5.0
)

farmer_query = (
    "What should I do to manage this disease in my crop? "
    "Please provide practical treatment and management guidance."
)

knowledge = knowledge_agent.retrieve(
    query=farmer_query,
    crop=diagnosis.crop,
    disease=diagnosis.disease,
    evidence_type="treatment",
    recommendation_preference=context.recommendation_preference,
    top_k=5,
)

print("Knowledge status :", knowledge.status)
print("Accepted evidence:", len(knowledge.accepted_evidence))
print("Rejected evidence:", len(knowledge.rejected_evidence))

for i, item in enumerate(knowledge.accepted_evidence, 1):
    print(
        f"  ACCEPTED {i}: "
        f"{item.knowledge_id} | "
        f"{item.metadata.get('recommendation_preference')} | "
        f"score={item.reranker_score:.4f}"
    )

for reason in knowledge.reasons:
    print("  REASON:", reason)

# ------------------------------------------------------------------
# 6. Grounded Generation
# ------------------------------------------------------------------
print("\n[6] Running Grounded Generation Agent...")

generation_agent = GroundedGenerationAgent(model)

image = Image.open(IMAGE_PATH).convert("RGB")

generation = generation_agent.generate(
    image=image,
    knowledge_result=knowledge,
    farmer_query=farmer_query,
    max_new_tokens=1536,
)

print("\n" + "=" * 70)
print("GROUNDED GENERATION RESULT")
print("=" * 70)

print("Status          :", generation.status)
print("Model called    :", generation.model_called)
print("Evidence count  :", generation.evidence_count)
print("Tokens          :", generation.generated_token_count)
print("Truncated       :", generation.truncated)
print("Inference time  :", generation.inference_time_seconds)

print("\n--- ANSWER ---")
print(generation.answer)

print("\n--- TRACEABILITY ---")
for item in generation.traceability:
    print(item)

print("\n" + "=" * 70)

if generation.abstain:
    print("RESULT: SAFE ABSTENTION")
    print("Reason: no sufficient eligible evidence reached generation.")
else:
    print("RESULT: GROUNDED GENERATION PASS")

print("=" * 70)

model.close()
