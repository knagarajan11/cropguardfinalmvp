# CropGuard Architecture Audit

## Audit basis

This is a static, read-only audit of the repository against
`CropGuard_Final_MVP_Architecture_v6_Chunking_Embedding_Aligned.md` and
`AGENTS.md`. No application code, model artifacts, datasets, packages, or
execution environment were changed. `claude3.py` is treated as the final
working evaluator and the primary runtime reference.

## A. ALREADY WORKING

| Component | Classification | Repository evidence |
|---|---|---|
| Current offline MVP | IMPLEMENTED | Image diagnosis/evaluation, LoRA utilities, and dataset/SFT preparation exist. |
| Nemotron Omni | IMPLEMENTED | `src/test_nemotron.py`, `claude3.py`, and `evaluate_cropguard1.py` load the local Nemotron Omni snapshot with `trust_remote_code=True`, BF16, and automatic device placement. |
| Validated LoRA | IMPLEMENTED | `claude3.py` loads the remapped LoRA with `PeftModel.from_pretrained(..., is_trainable=False)`; `create_remapped_lora.py` and `validate_remapped_lora.py` implement the 116-module/232-tensor remap and validation workflow. |
| Multimodal processor flow | IMPLEMENTED | `claude3.py` creates a chat template with image and text, uses `AutoProcessor`, validates image-token handling, and calls generation with processed image inputs. |
| Training image collation | IMPLEMENTED | `cropguard_collator.cropguard_nemotron_collate_fn()` loads images as RGB PIL images and resizes them to 512x512 before NVIDIA's Omni collator. |

Current demonstrable path:

`crop image -> Nemotron Omni + remapped CropGuard LoRA -> diagnosis/recommendation text -> offline evaluator metrics`

`claude3.py` is the final working evaluator. It is the reference to reuse for model loading, the safe in-memory generation patch, multimodal image handling, prompt construction, inference behavior, response parsing, and vision evaluation.

The known asset paths are:

- Base model: `/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2`
- Validated remapped LoRA: `/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped`

Committed evaluation evidence is limited but useful as a regression baseline:

- `cropguard_evaluation_results.csv`: 4 of 6 correct.
- `cropguard_base_vs_lora_results.csv`: base 5 of 6 correct; LoRA 4 of 6 correct, including one early-blight regression.
- `cropguard_evaluation_result.json`: one early-blight image was predicted as late blight.

## B. PARTIALLY IMPLEMENTED

| Component | Classification | Existing code and gap |
|---|---|---|
| End-to-end demo | PARTIALLY IMPLEMENTED | The CLI/offline diagnosis flow can be shown, but the blueprint's UI, context, RAG, evidence, memory, and multilingual flow cannot. |
| Diagnosis Agent | PARTIALLY IMPLEMENTED | `claude3.py` provides the underlying visual diagnosis through prompt-driven free text. It is not a separate, typed agent returning structured diagnosis signals. |
| PlantVillage / PlantDoc / Rice | PARTIALLY IMPLEMENTED | Download, metadata, normalization, splitting, validation, and SFT tools exist, but current local data is ignored and absent from this checkout. |
| Recommendation preference | PARTIALLY IMPLEMENTED | Prompts mention organic recommendations, but no Organic/IPM/General request attribute, metadata filter, or recommendation policy exists. |
| Answer parsing / technical validation | PARTIALLY IMPLEMENTED | `claude3.py` parses labels and validates the multimodal path; `evaluate_cropguard_all.py` contains `extract_disease()`, `labels_match()`, and `calculate_confidence()`. No grounded claim validator or abstention policy exists. |
| Vision evaluation | PARTIALLY IMPLEMENTED | Existing evaluators calculate classification metrics and compare base vs LoRA, but there is no retrieval, grounding, multilingual, or end-to-end evaluation. |

Dataset code worth reusing rather than rewriting:

- `datasets/download_plantvillage.py`, `datasets/download_plantdoc.py`, and `datasets/download_rice1426.py`
- `process_directory()` in each `datasets/prepare_*.py`
- `normalize_label()` and `process_file()` in `datasets/normalize_labels.py`
- `create_sft_record()` in `src/prepare_sft_dataset_0709.py`

`datasets/normalize_labels.py` expects `configs/label_mapping.yaml`, but that mapping file is not committed.

## C. MISSING

| Blueprint component | Classification | Audit result |
|---|---|---|
| Orchestrator / Supervisor | MISSING | No workflow owner receives a request and calls Diagnosis, Context, and Knowledge agents in order. |
| Context Agent | MISSING | No shared context for language, location, soil, weather, season, crop stage, prior actions, preference, or case state. |
| Knowledge Agent | MISSING | No query routing, source selection, metadata filters, hybrid retrieval, reranking, evidence packaging, or preference-aware recommendations. |
| RAG ingestion | MISSING | Dataset preparation is not canonical document ingestion. No common loader, document schema, or plug-and-play source configuration exists. |
| Token-based chunking | MISSING | No `TextSplitter`, structure-aware splitter, or 650/100/800/120 token policy exists. |
| NVIDIA embeddings | MISSING | No NVIDIA `llama-nemotron-embed-1b-v2`, passage/query types, or 2048-dimensional vector generation exists. |
| Vector store | MISSING | No Milvus/cuVS or lightweight vector-store provider exists. |
| BM25 / hybrid retrieval | MISSING | No sparse index, dense retrieval, fusion, or candidate selection exists. |
| NVIDIA reranking | MISSING | No reranker adapter or top-evidence selection exists. |
| Evidence Filter | MISSING | No evidence score threshold, treatment-vs-diagnosis separation, citation package, or eligibility checks exist. |
| Grounded generation | MISSING | Existing prompts generate advice without retrieved and cited evidence. |
| Answer validation / safe abstention | MISSING | No validation of treatment support, preference, language, region, dosage, or groundedness; no weak/conflicting-evidence abstention path. |
| Memory / case isolation | MISSING | CSV resumption in evaluators is not farmer memory. No current-case or persistent-profile store exists. |
| Soil / weather context | MISSING | No structured input, provider, caching, or retrieval metadata filters exist. |
| 10-language support | MISSING | No language configuration, translations, dynamic UI labels, or selected-language final generation exists. |
| FastAPI | MISSING | No routes, schemas, health endpoint, or independent API layer exists. |
| Streamlit | MISSING | No upload UI, session state, farmer controls, or result/evidence display exists. |
| Optional PostgreSQL / Redis | MISSING | No persistence abstraction, case store, farmer profile store, or cache exists. |
| Hackathon demonstration flow | MISSING | The Tamil/location/organic/image/context/RAG/evidence/follow-up journey cannot currently be demonstrated. |
| Deployment | MISSING | No launcher, deployment configuration, API/UI process topology, or tunnel separation exists. |

## D. FILES TO MODIFY

No existing validated evaluator or model utility should be modified as the first implementation step. Add the blueprint's modular application layer around the working reference instead.

| Requirement | Reuse | Files to add or modify |
|---|---|---|
| Vision boundary | `claude3.py` | Add `vision/nemotron_omni.py` and `vision/lora_loader.py`; do not import `claude3.py`, because it performs work at module scope. |
| Shared configuration | Existing YAML conventions | Add `app/config.py`, `config/settings.yaml`, `config/sources.yaml`, and `config/languages.yaml`. |
| Three-agent workflow | `claude3.py` diagnosis behavior | Add `agents/orchestrator.py`, `agents/diagnosis_agent.py`, `agents/context_agent.py`, and `agents/knowledge_agent.py`. |
| Context and memory | None | Add `context/` and `memory/` modules with a case-scoped store first. |
| RAG | Existing dataset-normalization concepts | Add `ingestion/`, `routing/`, `retrieval/`, and `validation/` modules. |
| API and UI | None | Add `app/main.py`, `api/routes.py`, `ui/streamlit_app.py`, and `ui/translations.py`. |
| Evaluation | Current vision result files and `claude3.py` | Add `evaluation/evaluate_vision.py`, `evaluation/evaluate_retrieval.py`, and `evaluation/evaluate_rag.py`. |
| Deployment | None | Add `.env.example`, `run.sh`, and deployment configuration without credentials. |

## E. FILES TO PRESERVE

- `AGENTS.md` and the architecture blueprint.
- All `configs/cropguard_nemotron_lora*.yaml` files.
- `cropguard_collator.py` and `cropguard_collator1.py`.
- `claude3.py` as the final working evaluator and canonical runtime reference.
- `src/test_nemotron.py`, `evaluate_cropguard1.py`, `test_remapped_inference.py`, and related tests as diagnostics/reference implementations.
- `create_remapped_lora.py`, `remap_lora_checkpoint.py`, `validate_remapped_lora.py`, and `test_remapped_lora.py`.
- `cropguard_evaluation_results.csv`, `cropguard_base_vs_lora_results.csv`, and `cropguard_evaluation_result.json`.
- All dataset scripts and the external ignored model cache, checkpoints, LoRA artifacts, outputs, datasets, `nemo_sandbox`, and `nemo-automodel.sif`.

## F. RECOMMENDED BUILD ORDER

1. Add settings and typed shared request/context/result schemas.
2. Isolate the final `claude3.py` vision path into a new `vision/nemotron_omni.py` module without modifying the evaluator.
3. Implement the structured Diagnosis Agent.
4. Add current-case memory and a Context Agent, with Redis/PostgreSQL optional.
5. Build FastAPI diagnose/chat/history/health boundaries.
6. Build the smallest Streamlit UI with image upload, location, language, preference, and follow-up chat.
7. Create canonical PlantVillage/PlantDoc/Rice knowledge records and common metadata.
8. Add structure-aware chunking, NVIDIA embedding adapter, vector store, BM25, hybrid fusion, and reranking.
9. Implement the Knowledge Agent, Evidence Filter, grounded generation, Answer Validator, and abstention path.
10. Add the ten required language resources and selected-language output enforcement.
11. Add retrieval, grounding, preference, multilingual, and full-flow regression tests.
12. Run only a small Slurm GPU validation using the existing sandbox after static and CPU-level checks pass.

## G. FIRST CHANGE TO MAKE

Create `vision/nemotron_omni.py` as a new, model-specific runtime adapter based on the final `claude3.py` evaluator.

It should preserve:

- configurable existing base-model and remapped-LoRA paths;
- RGB PIL conversion and processor chat-template flow;
- BF16, evaluation mode, and `is_trainable=False`;
- the in-memory inner-generation metadata patch;
- a narrow image-plus-prompt diagnosis interface.

It must not retrain, merge, copy, remap, save, or modify adapter weights at runtime.

## H. RISKS

- Preserve the original adapter unchanged. `remap_lora_checkpoint.py` is an artifact-writing utility and should not be run during application startup.
- The remapped adapter's `language_model.model.layers` to `language_model.backbone.layers` mapping is essential. Preserve the expected 116 modules and 232 tensors.
- Preserve the `claude3.py` safe in-memory generation behavior. The outer multimodal wrapper requires image metadata, but its inner language model rejects `num_patches`, `num_tokens`, and `imgs_sizes`.
- Do not import current evaluator files in production modules: they execute model loading and inference-oriented work at top level.
- Preserve RGB image conversion and processor chat templates. Text-only calls are not validated vision inference.
- Do not load duplicate 30B base models; use one base model with a LoRA overlay.
- Do not expose hidden reasoning. Existing evaluation output can contain reasoning markers; the final app must expose only validated farmer-facing content.
- The six-image evaluation is too small and contains a LoRA regression. Do not claim high real-world diagnostic or treatment reliability from it.
- The current checkout excludes data, checkpoints, outputs, evaluation images, and model cache; they are external DGX assets and must be preserved.
- Follow `AGENTS.md`: use the existing Apptainer environment, obtain a Slurm GPU allocation for GPU work, and do not alter CUDA, PyTorch, Transformers, NeMo, `nemo_sandbox`, or the `.sif`.
