# CropGuard — Final Hackathon MVP Architecture & Implementation Blueprint

## 1. Purpose & Final MVP Scope

CropGuard is a multimodal agricultural AI assistant for a hackathon demonstration on an NVIDIA DGX server. It combines:

- Plant-image diagnosis
- Farmer conversation and case memory
- Soil and weather context
- Targeted RAG
- NVIDIA-aligned retrieval and reranking
- Grounded answer generation
- Answer validation and safe abstention
- Multilingual farmer interaction

The current MVP uses the three available datasets — **PlantVillage, PlantDoc, and Rice** — and the already completed **Nemotron 3 Nano Omni 30B A3B Reasoning LoRA** fine-tuning based on those datasets.

The final MVP must reuse the validated LoRA. It must not retrain the model during application startup.

The RAG and ingestion layers must remain plug-and-play so future sources such as TNAU, ICAR, FAO, research papers, additional crops, and regional advisories can be added without redesigning the core application.

---

## 2. Final Recommended Architecture

```text
                              CROPGUARD
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
              Image          Farmer Query      UI Context
                                                Language
                                                Location
                                                Preference
                 │                │                │
                 └────────────────┼────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ ORCHESTRATOR / SUPERVISOR │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
 ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │ DIAGNOSIS AGENT │     │ CONTEXT AGENT    │     │ KNOWLEDGE AGENT  │
 └────────┬────────┘     └────────┬─────────┘     └────────┬─────────┘
          │                       │                        │
 Nemotron Omni + LoRA                                 NVIDIA Retriever
 PlantVillage                                         Embeddings
 PlantDoc                   Soil                     Hybrid Retrieval
 Rice                       Weather                  Reranker
                            Region                   Evidence
                            Season                   Organic / IPM / General Recommendations
                            Crop Stage
                            Previous Actions
          │                       │                        │
          └───────────────────────┼────────────────────────┘
                                  ▼
                                │
                                ▼
                       ┌──────────────────┐
                       │ EVIDENCE FILTER  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ NEMOTRON / NIM   │
                       │ GROUNDED         │
                       │ GENERATION       │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ ANSWER VALIDATOR │
                       └────────┬─────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
                  Grounded             Uncertain
                   Answer              / Abstain
                     │
                     ▼
                Memory Update
                    |
                    ▼
                    Receive and Record Customer Feedback  
```

### Core principle

```text
Diagnosis
   ↓
Context
   ↓
Knowledge Agent:
Query Understanding / Routing
   ↓
Retrieval
   ↓
Reranking
   ↓
Evidence Filtering
   ↓
Grounded Generation
   ↓
Answer Validation
   ↓
Multilingual Response
   ↓
Memory Update
   ↓
  Receive and Record Customer Feedback
```

---

## 3. Current Assets and Non-Negotiable Reuse Rules

### 3.1 Completed Nemotron Omni LoRA

The completed CropGuard LoRA was tuned using:

- PlantVillage
- PlantDoc
- Rice

The final MVP must load and reuse the validated adapter through configuration.

Example:

```text
BASE_MODEL_PATH=<existing Nemotron Omni model/cache>
LORA_ADAPTER_PATH=<validated CropGuard LoRA adapter>
```

Known validated assets:

```text
Base model:
 /workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2

Validated remapped LoRA:
 /workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/epoch_1_step_26459/model_remapped
```

The original adapter must remain untouched.

The application must not copy, modify, or retrain the adapter at runtime.

### 3.2 Multimodal inference requirement

The final MVP must preserve the validated Nemotron multimodal processor flow.

The processor must construct the multimodal conversation using the image content and text prompt before generating.

Do not directly pass processor-only metadata such as:

```text
num_patches
num_tokens
imgs_sizes
```

to an underlying language-model `generate()` call unless the actual Nemotron wrapper explicitly consumes those fields.

The final implementation must isolate this model-specific behavior inside:

```text
vision/nemotron_omni.py
```

so the rest of CropGuard remains model-agnostic.

---

# 4. Multilingual Farmer Experience

## 4.1 Supported languages

The final MVP must provide exactly these initial language options:

| Code | Language | Native display |
|---|---|---|
| `en` | English | English |
| `ta` | Tamil | தமிழ் |
| `kn` | Kannada | ಕನ್ನಡ |
| `te` | Telugu | తెలుగు |
| `hi` | Hindi | हिन्दी |
| `ml` | Malayalam | മലയാളം |
| `mr` | Marathi | मराठी |
| `or` | Odia | ଓଡ଼ିଆ |
| `pa` | Punjabi | ਪੰਜਾਬੀ |
| `bn` | Bengali | বাংলা |

The architecture must allow additional languages to be added without modifying the diagnosis, retrieval, or orchestration logic.

## 4.2 Language is a first-class request attribute

Language must not be treated as only a front-end display setting.

The shared request/context object should contain:

```json
{
  "language": "ta",
  "location": "Tamil Nadu",
  "recommendation_preference": "organic",
  "farmer_query": "...",
  "image": "..."
}
```

The selected language must be available to the orchestrator, context agent, Knowledge Agent, and final generation layer.

## 4.3 Dynamic Streamlit UI

Streamlit is the recommended final MVP UI.

The language selector should remain compact:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 🌱 CropGuard                  📍 Tamil Nadu │ 🗣 தமிழ் │ 🌿 Organic │
└──────────────────────────────────────────────────────────────────────┘
```

Changing the language must dynamically update user-facing UI text.

Examples:

```text
English:
Upload Crop Image
Enter your question
Analyze Crop
Diagnosis
Confidence
Recommendation
Evidence
Farm Context

Tamil:
பயிர் படத்தை பதிவேற்றவும்
உங்கள் கேள்வியை உள்ளிடவும்
பயிரை ஆய்வு செய்யவும்
நோய் கண்டறிதல்
நம்பகத்தன்மை
பரிந்துரை
ஆதாரம்
பண்ணை சூழல்
```

The same pattern must be implemented for all ten supported languages.

The language selector should be stored in Streamlit session state so it remains consistent throughout the farmer's conversation.

## 4.4 Dynamic input key and placeholder

When the farmer selects a language, the input label, placeholder, examples, and helper text must change to that language.

For example:

```text
English:
"Describe the problem with your crop..."

Tamil:
"உங்கள் பயிரில் உள்ள பிரச்சினையை விவரிக்கவும்..."
```

The application should use a centralized translation resource rather than scattering translated strings throughout the UI code.

Recommended:

```text
config/
  languages.yaml
```

or:

```text
ui/
  translations.py
```

## 4.5 AI response language

The final AI response must be generated in the selected language.

Example:

```text
Selected language = Tamil

Nemotron generation instruction:

Preferred response language: Tamil.
Respond entirely in Tamil.
Use scientifically important disease names in English
when a clear local equivalent is not available.
```

The language instruction must be applied at the final generation layer.

## 4.6 Retrieval language strategy

Do not require the current knowledge base to contain ten translated copies of every document.

The preferred flow is:

```text
Farmer Query
     ↓
Language = Tamil
     ↓
Canonical query representation
     ↓
RAG retrieval
     ↓
Evidence
     ↓
Nemotron
     ↓
Tamil response
```

The current PlantVillage, PlantDoc, and Rice knowledge can remain in its canonical indexed representation.

This separates:

```text
Retrieval language
```

from:

```text
Farmer response language
```

and keeps the knowledge layer scalable.

---

# 5. Compact Farmer Controls

The top-level UI should contain only the controls that materially affect the current interaction.

Recommended header:

```text
🌱 CropGuard       📍 Location ▼    🗣 Language ▼    🌿 Preference ▼
```

### Required controls

1. Location
2. Preferred language
3. Recommendation preference

Recommendation preference:

```text
Organic
IPM
General
```

These values must be part of the backend context.

They must not be UI-only fields.

### Farm context

Soil and weather should not occupy a large settings panel.

Use a compact expandable component:

```text
🌦 Farm Context ▾
29°C • 84% humidity • Recent rain • Clay loam • pH 6.8
```

Expanded view can contain:

- Soil type
- pH
- N/P/K
- Soil moisture
- Temperature
- Humidity
- Rainfall
- Leaf wetness
- Crop stage
- Region
- Season

---

# 6. Memory and Conversation Continuity

Conversation continuity is a first-class CropGuard capability.

Subsequent farmer questions must resolve relevant context from the current case without requiring the farmer to repeat information.

Example:

```text
Q1:
My tomato leaves have brown spots.

→ crop = tomato
→ candidate disease = early_blight

Q2:
What should I do organically?

→ tomato
→ early_blight
→ preference = organic

Q3:
I already removed the affected leaves. What next?

→ tomato
→ early_blight
→ organic
→ previous_action = removed leaves

Q4:
It rained heavily yesterday.

→ update weather context
→ preserve same case
```

## 6.1 Two memory layers

### Current case memory

Stores:

- Current conversation turns
- Current image
- Diagnosis
- Symptoms
- Recommendations
- Actions taken
- Context
- Feedback
- Language
- Location
- Recommendation preference

### Persistent farmer profile

Stores explicitly provided long-lived information:

- Region
- Farming preference
- Stable farm characteristics
- Stable soil characteristics when appropriate

### Information priority

```text
New user input
      ↓
Current image / diagnosis
      ↓
Current conversation memory
      ↓
Persistent farmer profile
      ↓
Defaults
```

A new crop/image must not inherit an unrelated previous diagnosis.

---

# 7. Agricultural Context

Soil and weather are structured contextual signals.

They should not be claimed as part of the LoRA training data unless actually used during training.

## Soil

Recommended fields:

```json
{
  "location": "...",
  "soil_type": "clay_loam",
  "ph": 6.8,
  "nitrogen": null,
  "phosphorus": null,
  "potassium": null,
  "organic_carbon": null,
  "moisture": null,
  "temperature": null,
  "salinity": null,
  "timestamp": null
}
```

## Weather

Recommended fields:

```json
{
  "location": "...",
  "timestamp": "...",
  "temperature": 29,
  "humidity": 84,
  "rainfall": 12,
  "wind_speed": null,
  "leaf_wetness": null,
  "forecast_rainfall": null
}
```

Weather and soil can influence:

- Context interpretation
- RAG filters
- Retrieval ranking
- Knowledge Agent recommendation preparation

---

# 8. Plug-and-Play Knowledge Architecture

Current sources:

```text
PlantVillage
PlantDoc
Rice
```

Future sources:

```text
TNAU
ICAR
FAO
Research papers
Regional agricultural advisories
Additional crop datasets
Government agricultural guidance
```

The retrieval engine must not contain source-specific logic such as:

```python
if source == "PlantVillage":
```

Instead:

```text
Source configuration
       ↓
Common loader
       ↓
Normalization
       ↓
Chunking
       ↓
Metadata
       ↓
Embedding
       ↓
Vector / Sparse Index
       ↓
Common Retrieval
       ↓
Reranking
```

Adding a new source should require:

```text
data
+
configuration
+
indexing
```

rather than rewriting the application.

## 8.2 Retrieval Ingestion Strategy — Chunking, Text Splitting and Embeddings

CropGuard should use a **structure-aware, token-based chunking strategy** rather than a simple fixed-character split. Agricultural knowledge contains diagnostic statements, treatment steps, tables, lists, disease descriptions, and region/crop-specific guidance. The splitter should preserve these semantic units so retrieval returns evidence that is complete enough to support safe recommendations.

### 8.2.1 Recommended chunking strategy

**Primary strategy: Structure-aware Recursive Token Splitter**

```text
Document
   ↓
Normalize
   ↓
Detect structure
(headings / paragraphs / lists / tables / captions)
   ↓
Create semantic sections
   ↓
Recursive token-based splitting
   ↓
Target ≈ 650 tokens/chunk
Overlap ≈ 100 tokens
   ↓
Attach agricultural metadata
   ↓
Embedding
   ↓
Vector index + BM25 index
```

Recommended defaults:

| Parameter | Recommended value | Reason |
|---|---:|---|
| Splitter | `RecursiveCharacterTextSplitter` with a model-aware token length function | Robust for mixed agricultural documents |
| Target chunk size | **650 tokens** | Focused evidence while retaining useful context |
| Chunk overlap | **100 tokens** | Preserves context across symptom → cause → management boundaries |
| Hard maximum | **800 tokens** | Prevents oversized chunks from reducing retrieval precision |
| Minimum useful chunk | **120 tokens** | Avoids indexing tiny fragments |
| Retrieval candidates | **40–50** | Strong recall for hybrid retrieval |
| Final reranked evidence | **5–8** | Focused generation context |

The implementation should use **token-based limits**, not character counts. The token length function should use the tokenizer associated with the deployed NVIDIA embedding model where practical.

### 8.2.2 Semantic boundary priority

The splitter should try to preserve boundaries in this order:

```text
1. Document / page boundary
2. Section / heading boundary
3. Paragraph boundary
4. List-item boundary
5. Sentence boundary
6. Clause boundary
7. Token limit
```

A treatment procedure should not be split between chunks if it can fit within the target size.

### 8.2.3 Agricultural atomic-evidence rule

A chunk should ideally answer **one retrieval purpose**:

```text
Diagnosis evidence
OR
Treatment / management evidence
OR
Organic evidence
OR
IPM evidence
OR
Crop-management evidence
OR
Regional/context evidence
```

Do not combine unrelated diseases or unrelated crops into one chunk.

Where a source contains a large section covering several diseases, create disease-level chunks whenever headings or semantic boundaries permit.

### 8.2.4 Metadata is part of retrieval

Every chunk must inherit document metadata and may add chunk-level metadata:

```json
{
  "document_id": "tnau_rice_organic_001",
  "chunk_id": "tnau_rice_organic_001_c03",
  "source": "tnau",
  "crop": "rice",
  "disease": null,
  "knowledge_type": "organic",
  "treatment_type": "botanical",
  "organic_eligible": true,
  "region": "Tamil Nadu",
  "crop_stage": "vegetative",
  "language": "en",
  "evidence_level": "high",
  "section": "Organic management",
  "chunk_index": 3,
  "text": "..."
}
```

Apply metadata filters before or during retrieval wherever supported:

```text
crop
disease
knowledge_type
treatment_type
organic_eligible
region
crop_stage
language
evidence_level
source
```

Semantic similarity alone must not decide whether evidence is eligible for the farmer's selected recommendation preference.

### 8.2.5 Tables and lists

Agricultural documents frequently contain treatment, symptom, crop-stage, dosage, and regional recommendation tables.

The ingestion pipeline should preserve row/column relationships:

```text
Table
 ↓
Preserve title + headers
 ↓
Convert to retrieval-friendly text
 ↓
Keep related row groups together
 ↓
Attach source/page/table metadata
```

Do not flatten a table into isolated cells.

Lists should preserve their numbering/bullet relationship when the list represents a procedure.

### 8.2.6 Optional parent-child context for long documents

For long TNAU/ICAR/FAO/research documents, the implementation may maintain:

```text
Parent section: 1,200–1,600 tokens
        │
        ├── Child chunk 1: ~650 tokens
        ├── Child chunk 2: ~650 tokens
        └── Child chunk 3: ~650 tokens
```

Search the child chunks, while optionally supplying the child plus parent-section context to final generation.

This is useful for long advisory documents. For the initial hackathon MVP, parent-child expansion is **optional**; the mandatory baseline is structure-aware ~650-token child chunking.

---

## 8.3 Recommended Embedding Model

### Primary text RAG embedding

For CropGuard's **textual knowledge base**, use:

```text
NVIDIA
llama-nemotron-embed-1b-v2

Model ID:
nvidia/llama-nemotron-embed-1b-v2
```

This model is designed for multilingual/cross-lingual question-answering retrieval and long-document retrieval, with up to 8192 input tokens and configurable embedding dimensions.

Recommended baseline:

```yaml
embedding:
  provider: nvidia
  model: nvidia/llama-nemotron-embed-1b-v2
  dimensions: 2048
  input_type_query: query
  input_type_document: passage
```

Use the full **2048-dimensional** representation initially because retrieval quality is more important than minimizing vector storage for the hackathon.

The model supports reduced dimensions when storage or latency becomes important. That should remain a deployment tuning parameter rather than an application redesign.

### Multimodal embedding option

If the knowledge base later contains retrieval-critical images, diagrams, scanned pages, or image+text evidence, use:

```text
nvidia/llama-nemotron-embed-vl-1b-v2
```

Keep this as an optional multimodal index rather than replacing the primary text embedding path.

```text
Text knowledge
     ↓
llama-nemotron-embed-1b-v2
     ↓
Text vector index

Image / image+text evidence
     ↓
llama-nemotron-embed-vl-1b-v2
     ↓
Multimodal vector index
```

For the current MVP, PlantVillage/PlantDoc/Rice images are primarily used by the **Diagnosis Agent**. The RAG knowledge layer should therefore use the text embedding path by default.

### Model responsibility separation

Do not use Nemotron Omni + CropGuard LoRA as the vector embedding service.

```text
Nemotron Omni + LoRA
    → image diagnosis / multimodal reasoning

NVIDIA Embed model
    → dense retrieval embeddings

BM25
    → exact lexical retrieval

NVIDIA Reranker
    → query-document relevance scoring

Nemotron / NIM
    → grounded final answer generation
```

This separation keeps diagnosis, retrieval and generation independently testable and replaceable.

---

## 8.4 Text Splitter Implementation Contract

The ingestion layer should expose:

```python
class TextSplitter:
    def split(self, document):
        ...
```

The CropGuard implementation should provide:

```text
StructureAwareTokenSplitter
```

with:

```yaml
chunking:
  strategy: structure_aware_recursive
  target_tokens: 650
  overlap_tokens: 100
  max_tokens: 800
  min_tokens: 120
  preserve_headings: true
  preserve_lists: true
  preserve_tables: true
  parent_child_context: optional
```

A practical implementation may use LangChain's recursive splitter with a custom tokenizer length function, but the splitter must remain behind the `ingestion/chunker.py` abstraction.

Recommended constants:

```python
CHUNK_TARGET_TOKENS = 650
CHUNK_OVERLAP_TOKENS = 100
CHUNK_MAX_TOKENS = 800
CHUNK_MIN_TOKENS = 120
```

Avoid a naïve configuration such as:

```python
chunk_size=1000
chunk_overlap=0
```

because it can separate important agricultural evidence and does not protect against semantic fragmentation.

---

## 8.5 Final CropGuard RAG Ingestion Pipeline

```text
Source
  ↓
Loader
  ↓
Content normalization
  ↓
Structure detection
  ├─ headings
  ├─ paragraphs
  ├─ lists
  ├─ tables
  └─ captions
  ↓
Structure-aware recursive token chunking
  ↓
~650-token chunks
~100-token overlap
  ↓
Metadata enrichment
  ↓
NVIDIA llama-nemotron-embed-1b-v2
  ↓
Vector Index
  +
BM25 Sparse Index
  ↓
Hybrid Fusion
  ↓
Top 40–50
  ↓
NVIDIA Reranker
  ↓
Top 5–8 evidence chunks
  ↓
Evidence Filter
  ↓
Nemotron Grounded Generation
```

For current PlantVillage, PlantDoc and Rice data:

```text
Dataset
 ↓
Canonical diagnostic/knowledge records
 ↓
Metadata normalization
 ↓
Structure-aware chunking
 ↓
NVIDIA text embeddings
 ↓
Vector + BM25 index
```

The image remains available to the Diagnosis Agent and does not need to be converted into a text-RAG embedding.

For future TNAU/ICAR/FAO/research documents:

```text
PDF/DOCX/HTML
 ↓
NV-Ingest / NeMo Retriever
 ↓
Layout / table / OCR extraction
 ↓
Structure-aware chunking
 ↓
NVIDIA embeddings
 ↓
Milvus + cuVS
 ↓
Hybrid retrieval
 ↓
NVIDIA reranking
```

The application-level retrieval interface remains unchanged.

---

## 8.6 Common metadata schema

```json
{
  "document_id": "pv_tomato_001",
  "text": "...",
  "source": "plantvillage",
  "crop": "tomato",
  "disease": "early_blight",
  "knowledge_type": "disease",
  "treatment_type": null,
  "organic_eligible": false,
  "region": "global",
  "crop_stage": null,
  "language": "en",
  "evidence_level": "dataset"
}
```

A future TNAU record can use the same structure:

```json
{
  "document_id": "tnau_rice_organic_001",
  "text": "...",
  "source": "tnau",
  "crop": "rice",
  "disease": null,
  "knowledge_type": "organic",
  "treatment_type": "botanical",
  "organic_eligible": true,
  "region": "Tamil Nadu",
  "crop_stage": "vegetative",
  "language": "en",
  "evidence_level": "high"
}
```

---

# 9. Recommendation Preference and Evidence Rules

The selected recommendation preference must affect retrieval.

## Organic

```text
organic_eligible = true
```

Only evidence supporting organic recommendations should be selected for treatment claims.

## IPM

Retrieve evidence covering applicable:

- Cultural management
- Mechanical management
- Biological management
- Botanical management
- IPM practices

## General

Use all approved evidence subject to the evidence and safety rules.

### Critical rule

Diagnosis evidence and treatment evidence are different.

PlantVillage, PlantDoc, and Rice may support diagnosis.

Specific treatment recommendations should be supported by appropriate management evidence when available.

The system must not infer an organic treatment merely because it knows the disease.

If sufficient treatment evidence is unavailable:

```text
DO NOT GUESS
```

Ask for more information or abstain.

---

# 10. Knowledge Agent — Query Understanding / Routing

Query understanding and routing are internal capabilities of the **Knowledge Agent**. The Knowledge Agent answers:

> What information do I need?

It prepares the retrieval plan, validates the requested scope and filters, and then invokes the retrieval capabilities. Routing is not a separate CropGuard agent.

Recommended output:

```json
{
  "intent": [
    "disease_diagnosis",
    "treatment_recommendation"
  ],
  "crop": "rice",
  "symptoms": [
    "brown_spots"
  ],
  "environment": {
    "heavy_rain": true
  },
  "treatment_preference": "organic",
  "language": "ta",
  "target_collections": [
    "disease_kb",
    "organic_kb",
    "crop_kb"
  ],
  "filters": {
    "crop": "rice"
  }
}
```

The application must validate the Knowledge Agent routing output against allow-lists before retrieval is executed.

---

# 11. Retrieval

Retrieval is an internal capability invoked by the **Knowledge Agent** after query understanding, routing, and metadata filtering.

Use hybrid retrieval:

```text
                     Query
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Dense Retrieval      Sparse/BM25
       semantic match      exact terms
              │                 │
              └────────┬────────┘
                       ▼
                     Fusion
                       ▼
                  Top 30–50
                       ▼
                   Reranker
                       ▼
                    Top 5–8
```

Dense retrieval uses `nvidia/llama-nemotron-embed-1b-v2` for the primary textual knowledge index.

Recommended dense embedding dimension:

```text
2048
```

Use the same model for:
- Query embeddings (`input_type=query`)
- Passage embeddings (`input_type=passage`)

The default chunking target is approximately **650 tokens with 100-token overlap**, using structure-aware recursive token splitting.

Sparse/BM25 is useful for:

- Disease names
- Scientific names
- Crop varieties
- Agricultural terminology
- Treatment names
- Source-specific terminology

---

# 12. NVIDIA Retrieval / Reranking Integration

The **Knowledge Agent** uses these retrieval capabilities through adapters. The application should isolate NVIDIA services behind adapters.

Recommended interfaces:

```text
VisionModel
EmbeddingModel
RerankerModel
GenerationModel
QueryUnderstandingModel
```

`QueryUnderstandingModel` is an internal capability used by the Knowledge Agent; it is not a separate CropGuard agent.

Model endpoints and API keys must be configuration-driven.

Recommended NVIDIA-aligned components:

```text
NeMo Retriever
NVIDIA embedding model
NVIDIA reranking NIM
Nemotron generation
```

Exact deployed model names and endpoints must be configurable because the DGX environment may expose local NIM services differently from hosted NVIDIA services.

---

# 13. Vector Database and Storage

Recommended NVIDIA-oriented option:

```text
Milvus + cuVS
```

Milvus can support:

- Disease KB
- Organic-management KB
- Crop-management KB
- Soil KB
- Semantic farmer history/events

The architecture should not make Milvus mandatory for every local demo configuration.

For a lightweight hackathon run, the implementation may provide a local/simple vector index provider behind the same interface.

---

# 14. Model Architecture

## Diagnosis

```text
Image + Farmer Query
        ↓
Nemotron Omni + CropGuard LoRA
        ↓
Diagnosis
```

The existing trained LoRA remains the primary CropGuard diagnosis component.

## Generation

```text
Diagnosis
+
Context
+
Ranked Evidence
+
Farmer Query
+
Language
+
Recommendation Preference
        ↓
Nemotron / NVIDIA NIM
        ↓
Grounded Response
```

The final response must not expose hidden chain-of-thought. The final generation layer receives the Evidence Filter output and generates only the farmer-facing grounded response.

Only concise reasoning summaries, evidence, confidence, and recommendations should be shown.

---

# 15. Answer Validation and Grounding

The final answer validator must check:

```text
✓ Diagnosis claims supported?
✓ Treatment claims supported?
✓ Organic/IPM preference respected?
✓ Crop consistent?
✓ Region consistent where relevant?
✓ Context consistent?
✓ Language correct?
✓ Evidence cited?
✓ Unsupported dosage avoided?
✓ Unsupported certainty avoided?
```

Strong evidence:

```text
Strong evidence
+
Consistent context
+
Grounding passes
        ↓
Generate answer
```

Weak/conflicting evidence:

```text
Weak evidence
OR
Conflicting evidence
        ↓
Do NOT invent certainty
        ↓
Ask for more information
or abstain
```

---

# 16. End-to-End Multilingual Example

## Farmer interaction

```text
Location: Tamil Nadu
Language: தமிழ்
Preference: Organic

Farmer:
"என் நெல் இலைகளில் பழுப்பு நிற புள்ளிகள் உள்ளன.
நேற்று அதிகமாக மழை பெய்தது. இயற்கை முறையில்
என்ன செய்யலாம்?"
```

## Step 1 — Diagnosis

```text
Crop: Rice

Observed symptom:
Brown leaf spots

Diagnosis candidate:
Disease

Confidence:
0.88
```

## Step 2 — Context

```text
Crop: rice
Region: Tamil Nadu
Crop stage: vegetative
Soil: clay
Weather: heavy rain
Treatment preference: organic
Language: Tamil
```

## Step 3 — Routing

```text
Target:
Disease KB
Organic KB
Crop KB
```

## Step 4 — Retrieval

```text
Dense + BM25
       ↓
Candidate evidence
       ↓
NVIDIA Reranker
       ↓
Top evidence
```

## Step 5 — Generation

Input to final generation:

```text
Farmer query
+
Diagnosis
+
Context
+
Ranked evidence
+
Preference = organic
+
Language = Tamil
```

Generation instruction:

```text
Respond entirely in Tamil.
Use only supported evidence.
Do not invent organic treatments.
Clearly indicate uncertainty when evidence is insufficient.
```

## Step 6 — Validation

Validate:

```text
✓ Rice
✓ Brown spots
✓ Organic preference
✓ Tamil response
✓ Treatment evidence
✓ Grounding
```

If treatment evidence is insufficient:

```text
The system must not invent a treatment.

It should explain that sufficient evidence is unavailable
and request additional information or recommend consulting
an appropriate agricultural expert.
```

---

# 17. Streamlit Final MVP UI

Recommended screen:

```text
┌───────────────────────────────────────────────────────────────────────┐
│ 🌱 CropGuard              📍 Tamil Nadu │ 🗣 தமிழ் │ 🌿 Organic      │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  📷 பயிர் படத்தை பதிவேற்றவும்                                         │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                         Image                                   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  💬 உங்கள் கேள்வியை உள்ளிடவும்                                       │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ உங்கள் பயிரில் உள்ள பிரச்சினையை விவரிக்கவும்...                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│                         [ 🔍 ஆய்வு செய்யவும் ]                        │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ 🌱 நோய் கண்டறிதல்                                               │  │
│  │ Rice — Brown Spot                                               │  │
│  │ Confidence: 88%                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  🌿 பரிந்துரை                                                        │
│  ...                                                                  │
│                                                                       │
│  🌦 Farm Context ▾                                                   │
│  29°C • 84% humidity • Recent rain • Clay loam • pH 6.8              │
│                                                                       │
│  📚 ஆதாரம் ▾                                                        │
│  PlantVillage • PlantDoc • Rice                                     │
│                                                                       │
│  ✓ Grounded response                                                 │
└───────────────────────────────────────────────────────────────────────┘
```

The UI should remain compact and demo-friendly.

Do not create a large settings panel consuming a significant portion of the screen.

---

# 18. Recommended Project Structure

The project structure mirrors the Final Recommended Architecture: one Orchestrator and three functional agents — Diagnosis, Context, and Knowledge. Routing, retrieval, reranking, evidence filtering, and recommendation preparation are internal modules used by the Knowledge Agent rather than separate agents.


```text
/workspace/finalmvp/
│
├── app/
│   ├── main.py
│   ├── config.py
│   └── dependencies.py
│
├── agents/
│   ├── orchestrator.py
│   ├── diagnosis_agent.py
│   ├── context_agent.py
│   └── knowledge_agent.py
│
├── vision/
│   ├── nemotron_omni.py
│   └── lora_loader.py
│
├── memory/
│   ├── conversation_memory.py
│   ├── farmer_profile.py
│   ├── case_manager.py
│   └── memory_store.py
│
├── context/
│   ├── context_builder.py
│   ├── soil.py
│   ├── weather.py
│   ├── region.py
│   └── season.py
│
├── ingestion/
│   ├── loaders.py
│   ├── normalizer.py
│   ├── chunker.py
│   ├── text_splitter.py
│   ├── metadata.py
│   └── pipeline.py
│
├── retrieval/
│   ├── retriever.py
│   ├── hybrid.py
│   ├── reranker.py
│   ├── vector_store.py
│   └── filters.py
│
├── routing/
│   ├── rag_router.py
│   └── schemas.py
│
│   # Routing is an internal Knowledge Agent capability; it is not a separate agent.
│
├── generation/
│   ├── nemotron.py
│   └── prompts.py
│
├── validation/
│   ├── evidence_filter.py
│   ├── grounding.py
│   └── answer_validator.py
│
├── api/
│   └── routes.py
│
├── ui/
│   ├── streamlit_app.py
│   └── translations.py
│
├── data/
│   ├── plantvillage/
│   ├── plantdoc/
│   └── rice/
│
├── knowledge_base/
│   ├── disease/
│   ├── organic/
│   ├── crop_management/
│   ├── tnau/
│   ├── icar/
│   ├── fao/
│   └── research/
│
├── indexes/
│
├── config/
│   ├── sources.yaml
│   ├── languages.yaml
│   └── settings.yaml
│
├── evaluation/
│   ├── evaluate_vision.py
│   ├── evaluate_retrieval.py
│   └── evaluate_rag.py
│
├── scripts/
│   ├── build_index.py
│   ├── ingest_source.py
│   └── health_check.py
│
├── tests/
│
├── logs/
│
├── requirements.txt
├── .env.example
├── Dockerfile
├── README.md
└── run.sh
```

---

# 19. Configuration

Example:

```yaml
languages:
  en:
    name: English
    native_name: English
  ta:
    name: Tamil
    native_name: தமிழ்
  kn:
    name: Kannada
    native_name: ಕನ್ನಡ
  te:
    name: Telugu
    native_name: తెలుగు
  hi:
    name: Hindi
    native_name: हिन्दी
  ml:
    name: Malayalam
    native_name: മലയാളം
  mr:
    name: Marathi
    native_name: मराठी
  or:
    name: Odia
    native_name: ଓଡ଼ିଆ
  pa:
    name: Punjabi
    native_name: ਪੰਜਾਬੀ
  bn:
    name: Bengali
    native_name: বাংলা
```

Sources:

```yaml
sources:
  plantvillage:
    enabled: true
    path: /workspace/finalmvp/data/plantvillage
    type: image_dataset
    category: disease
    priority: 1

  plantdoc:
    enabled: true
    path: /workspace/finalmvp/data/plantdoc
    type: image_dataset
    category: disease
    priority: 1

  rice:
    enabled: true
    path: /workspace/finalmvp/data/rice
    type: image_dataset
    category: disease
    priority: 1

  tnau:
    enabled: false
    path: /workspace/finalmvp/knowledge_base/tnau
    type: document
    category: organic
    priority: 2

  icar:
    enabled: false
    path: /workspace/finalmvp/knowledge_base/icar
    type: document
    category: organic
    priority: 2

  fao:
    enabled: false
    path: /workspace/finalmvp/knowledge_base/fao
    type: document
    category: organic
    priority: 2

  research:
    enabled: false
    path: /workspace/finalmvp/knowledge_base/research
    type: document
    category: research
    priority: 3
```

---

# 20. Environment Configuration

Do not hard-code deployment-specific values.

Example:

```text
BASE_MODEL_PATH=
LORA_ADAPTER_PATH=

NVIDIA_API_KEY=

VISION_NIM_URL=
EMBED_NIM_URL=
RERANK_NIM_URL=
GENERATION_NIM_URL=

VECTOR_DB_HOST=
VECTOR_DB_PORT=

DATABASE_URL=

REDIS_URL=

DEMO_MODE=false
USE_MCP=false

TOP_K_RETRIEVAL=40
TOP_K_RERANK=8
MIN_EVIDENCE_SCORE=0.70

EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2
EMBEDDING_DIMENSIONS=2048
EMBEDDING_QUERY_INPUT_TYPE=query
EMBEDDING_PASSAGE_INPUT_TYPE=passage

CHUNKING_STRATEGY=structure_aware_recursive
CHUNK_TARGET_TOKENS=650
CHUNK_OVERLAP_TOKENS=100
CHUNK_MAX_TOKENS=800
CHUNK_MIN_TOKENS=120
```

Secrets must never be committed to source control.

---

# 21. API Layer

Recommended endpoints:

```text
POST /api/v1/diagnose
POST /api/v1/chat
GET  /api/v1/farmer/{farmer_id}/history
GET  /api/v1/health
GET  /api/v1/metrics
```

Example diagnosis request:

```json
{
  "language": "ta",
  "location": "Tamil Nadu",
  "recommendation_preference": "organic",
  "question": "...",
  "image": "..."
}
```

The API must remain independent from Streamlit.

Streamlit is the presentation layer, not the business-logic layer.

---

# 22. PostgreSQL and Redis

## PostgreSQL

Use PostgreSQL for durable structured information:

- Farmer profile
- Location
- Crop
- Soil profile
- Crop stage
- Farming preference
- Diagnosis records
- Actions
- Feedback
- Case metadata

## Redis

Redis is optional.

Use it for:

- Current session state
- Short-lived conversation context
- Weather/API cache
- Retrieval-result cache
- Short-TTL information

The MVP must remain runnable without Redis if caching is not required.

---

# 23. Evaluation

Evaluate both retrieval and generation.

## Retrieval

- Context Recall
- Context Precision
- Context Relevance
- Context Entity Recall

## Answer / Grounding

- Faithfulness
- Response Groundedness
- Response Relevancy
- Noise Sensitivity

## Multilingual evaluation

Add checks for:

- Correct selected language
- No unintended language switching
- Preservation of disease/scientific terminology
- Recommendation preference preservation
- Grounding after translation/generation
- Response readability

---

# 24. Reliability and Safety

The final hackathon build must prioritize:

```text
WORKING > COMPLEX
RELIABLE > OVER-ENGINEERED
DEMONSTRABLE > THEORETICAL
MODULAR > HARD-CODED
```

If an NVIDIA service is unavailable, the application should fail gracefully and clearly report the unavailable component.

Demo fallbacks may be provided, but they must be clearly marked as demo components.

The system must never fabricate:

- Disease certainty
- Treatment
- Dosage
- Source
- Evidence
- Weather information
- Soil information
- Confidence

Agricultural recommendations should be grounded in retrieved evidence.

---

# 25. NVIDIA Enterprise RAG Alignment

CropGuard can evolve toward an NVIDIA Enterprise RAG architecture:

```text
Documents
   ↓
NV-Ingest / NeMo Retriever
   ↓
Extraction + Metadata
   ↓
Embedding
   ↓
Vector Database
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Evidence Filtering
   ↓
Nemotron Generation
   ↓
Guardrails
   ↓
RAG Evaluation
   ↓
Observability
```

The complete enterprise stack should not be deployed solely for the hackathon if it adds unnecessary operational complexity.

The enterprise blueprint is the target architecture.

The final MVP should implement the components that provide the strongest demonstrable value on the available DGX environment.

---

# 26. What Is and Is Not in LoRA Training

### Used in LoRA training

```text
PlantVillage
PlantDoc
Rice
```

### Not claimed as LoRA training data

```text
Soil
Weather
Farmer memory
Location
Language preference
Recommendation preference
TNAU
ICAR
FAO
Research papers
```

These are runtime/context/RAG capabilities.

This distinction must be preserved in the hackathon presentation.

---

# 27. Agent Responsibilities

The final architecture therefore has one supervisor and three functional agents:
**Diagnosis Agent, Context Agent, and Knowledge Agent**. Evidence filtering, generation,
answer validation, memory update, and feedback are workflow components around those agents,
not additional agents.

The final architecture contains **three functional agents under the Orchestrator / Supervisor**. Recommendation generation is a responsibility of the **Knowledge Agent**, not a separate agent. This keeps the runtime simple and matches the Final Recommended Architecture.

## Orchestrator / Supervisor

Responsible for:

- Receiving the image, farmer query, and UI context
- Maintaining the workflow order
- Invoking Diagnosis, Context, and Knowledge agents
- Passing shared context between agents
- Handling failures and unavailable services
- Sending the final evidence package to the generation layer
- Returning the validated response and memory-update event

## Diagnosis Agent

Responsible for:

- Image analysis
- Crop/disease diagnosis
- Confidence
- Diagnostic signals
- Returning a structured diagnosis for downstream context and knowledge processing

Uses:

```text
Nemotron Omni + CropGuard LoRA
```

## Context Agent

Responsible for:

- Farmer memory
- Case memory
- Location
- Soil
- Weather
- Season
- Crop stage
- Previous actions
- Farmer preference
- Language
- Producing the shared context used by the Knowledge Agent

## Knowledge Agent

Responsible for the complete knowledge and recommendation path shown in the Final Recommended Architecture:

- Query understanding
- Query routing
- Collection/knowledge-source selection
- Metadata filtering
- Hybrid retrieval
- Reranking
- Evidence packaging
- Organic/IPM/General recommendation retrieval
- Preparing a grounded recommendation from retrieved evidence
- Applying evidence and preference constraints before final generation

The Knowledge Agent may use internal routing, retrieval, reranking, and filtering modules, but these are **capabilities/components, not additional agents**.

The output passed to the Evidence Filter should contain the diagnosis, relevant context, ranked evidence, recommendation candidates, and grounding metadata required for final generation.

# 28. Final Hackathon Demonstration Flow

The ideal live demonstration should show:

```text
1. Open CropGuard

2. Select:
   Location = Tamil Nadu
   Language = தமிழ்
   Preference = Organic

3. UI immediately changes to Tamil

4. Upload crop image

5. Ask a Tamil question

6. Diagnosis Agent analyzes image

7. Context Agent retrieves:
   - Current case
   - Previous farmer interaction
   - Soil
   - Weather
   - Region

8. Knowledge Agent:
   - Understands query
   - Filters evidence
   - Performs hybrid retrieval
   - Reranks evidence

9. Knowledge Agent prepares recommendation

10. Evidence Filter checks recommendation

11. Nemotron generates Tamil response

12. Answer Validator checks grounding

13. UI displays:
    - Diagnosis
    - Confidence
    - Recommendation
    - Farm context
    - Evidence
    - Grounding status

14. Memory Update stores the interaction

15. Ask a follow-up question without repeating the crop/problem

16. CropGuard uses previous conversation context
```

---

# 29. Final Architecture Principle

```text
MULTIMODAL DIAGNOSIS
        +
CONVERSATIONAL MEMORY
        +
AGRICULTURAL CONTEXT
(SOIL + WEATHER + REGION)
        +
PLUG-AND-PLAY KNOWLEDGE
        +
HYBRID RAG
        +
NVIDIA RETRIEVAL / RERANKING
        +
GROUNDED NEMOTRON GENERATION
        +
MULTILINGUAL FARMER EXPERIENCE
        +
ANSWER VALIDATION
        +
SAFE ABSTENTION
```

The final CropGuard MVP should demonstrate a clear path from:

```text
Image + Farmer Question
        ↓
AI Diagnosis
        ↓
Context + Memory
        ↓
Evidence Retrieval
        ↓
Grounded Recommendation
        ↓
Selected Indian Language
        ↓
Validated Farmer Answer
        ↓
Memory Update
```

The architecture is intentionally modular so the hackathon MVP can remain reliable while evolving toward an enterprise-grade NVIDIA RAG implementation.

---

# 30. DGX External Access & Deployment

The final hackathon MVP runs entirely on the NVIDIA DGX. Streamlit is the external presentation layer but remains hosted on the DGX at `localhost:8501`. Because the DGX is behind a guarded firewall, external access is provided through a Cloudflare Quick Tunnel using an outbound `cloudflared` connection.

## 30.1 Recommended MVP topology

```text
                         INTERNET
                            │
                            ▼
                 Cloudflare Edge
                            │
                    Quick Tunnel
                            │
                            ▼
              cloudflared on DGX
                            │
                            ▼
                   Streamlit :8501
                            │
                    internal only
                            ▼
                    FastAPI :8000
                            │
                            ▼
                 CropGuard Core
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
     Nemotron+LoRA       RAG/Context       Memory
```

External users never connect directly to the DGX IP. The `cloudflared` process establishes the tunnel from the DGX to Cloudflare and routes the public URL to the local Streamlit service.

## 30.2 Hackathon access mode — Quick Tunnel

The MVP deliberately uses the simplest external-access mechanism. Start Streamlit locally and expose it with:

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8501
```

Cloudflare generates a temporary public `trycloudflare.com` URL for that run. The URL may change whenever the tunnel is restarted. This is intentional for the hackathon MVP and avoids adding domain, DNS, account, certificate, or persistent-tunnel infrastructure to the critical path.

## 30.3 Future permanent URL — configuration only

The deployment design must keep external-access mode separate from application code. The application must not contain Cloudflare-specific logic. Use a deployment configuration file such as `config/deployment.yaml`:

```yaml
tunnel:
  enabled: true
  mode: quick

  quick:
    target: http://localhost:8501

  named:
    enabled: false
    hostname: ""
    tunnel_name: ""
```

For a future organization-approved permanent URL, the intended change is configuration-only:

```yaml
tunnel:
  enabled: true
  mode: named

  named:
    enabled: true
    hostname: cropguard.example.com
    tunnel_name: cropguard-dgx
```

No changes should be required to Streamlit, FastAPI, the orchestrator, agents, RAG, Nemotron, or the LoRA. The exact Named Tunnel setup remains dependent on organizational Cloudflare/firewall policy.

## 30.4 Security and firewall rules

- Keep Streamlit bound to the DGX host; do not expose the DGX IP directly to the Internet.
- Keep FastAPI on `127.0.0.1:8000` or another DGX-internal interface.
- Do not expose the FastAPI port publicly for the hackathon.
- Do not put model credentials, NVIDIA API keys, or internal paths into the Streamlit client/UI.
- Apply image upload size/type limits and request timeouts.
- Log application errors and request status without logging sensitive farmer data unnecessarily.
- Use only outbound connectivity permitted by the DGX organization's firewall policy.
- For production deployment, replace the Quick Tunnel with an organization-approved ingress, gateway, VPN, or persistent tunnel.

## 30.5 Internal API boundary

The API layer remains useful even though it is not Internet-facing in the MVP. Recommended internal endpoints:

```text
POST /api/v1/diagnose
POST /api/v1/chat
GET  /api/v1/farmer/{farmer_id}/history
GET  /api/v1/health
GET  /api/v1/metrics
```

Streamlit calls these APIs locally. A future mobile/web client can use the same service boundary when an approved external API gateway is introduced.

## 30.6 Final startup sequence

The preferred final MVP launcher should keep startup simple:

```text
1. Load configuration
2. Start FastAPI on localhost:8000
3. Start Streamlit on localhost:8501
4. Start cloudflared Quick Tunnel
5. Capture/display the generated public URL
6. Judge opens the displayed URL
```

A single `run.sh` may orchestrate these processes so the demo operator does not need to remember multiple commands.

## 30.7 Deployment principle

```text
APPLICATION CODE
      │
      ├── Streamlit
      ├── FastAPI
      ├── CropGuard Core
      ├── RAG
      ├── Memory
      └── Nemotron + LoRA

EXTERNAL ACCESS = DEPLOYMENT CONFIGURATION
      │
      ├── Quick Tunnel  → random temporary URL (MVP)
      └── Named Tunnel   → permanent URL (future)
```

This preserves a low-complexity hackathon deployment while keeping the permanent-URL option easy to introduce later without application-code changes.

---

# 31. Final Implementation Rules

1. **Reuse the completed Nemotron Omni LoRA.**
2. **Do not retrain LoRA during MVP startup.**
3. **Keep base-model and LoRA paths configurable.**
4. **Preserve the validated multimodal processor/generation flow.**
5. **Keep PlantVillage, PlantDoc, and Rice configuration-driven.**
6. **Do not add source-specific branches to the RAG engine.**
7. **Keep soil and weather as structured context.**
8. **Make conversation/case memory first-class.**
9. **Make language a first-class context attribute.**
10. **Support English, Tamil, Kannada, Telugu, Hindi, Malayalam, Marathi, Odia, Punjabi, and Bengali.**
11. **Dynamically translate Streamlit labels, placeholders, buttons, messages, and response headings based on selected language.**
12. **Generate the final farmer-facing response in the selected language.**
13. **Keep retrieval independent of the response language where practical.**
14. **Make location affect context and future regional retrieval.**
15. **Make Organic/IPM/General a real Knowledge Agent retrieval/recommendation preference.**
16. **Separate diagnosis evidence from treatment evidence.**
17. **Do not invent treatment recommendations when evidence is insufficient.**
18. **Abstain when evidence is weak or conflicting.**
19. **Expose evidence and grounding status in the UI.**
20. **Implement only the three functional agents shown in the Final Recommended Architecture: Diagnosis, Context, and Knowledge.**
21. **Keep Streamlit as the presentation layer.**
22. **Keep the backend independently testable.**
23. **Keep PostgreSQL and Redis optional where practical.**
24. **Keep NVIDIA service endpoints configurable.**
25. **Do not expose hidden chain-of-thought.**
26. **Prioritize working, reliable, demonstrable functionality over unnecessary infrastructure.**

---

---

## Final RAG Defaults — Hackathon Baseline

```text
Chunking:
  Structure-aware recursive token splitter
  Target: 650 tokens
  Overlap: 100 tokens
  Hard maximum: 800 tokens
  Minimum useful chunk: 120 tokens

Dense embedding:
  NVIDIA nvidia/llama-nemotron-embed-1b-v2
  Dimension: 2048
  Query input_type: query
  Passage input_type: passage

Sparse retrieval:
  BM25

Hybrid retrieval:
  Dense + BM25
  Candidate pool: 40–50

Reranking:
  NVIDIA reranker
  Final evidence: 5–8 chunks

Generation:
  Nemotron / NVIDIA NIM

Safety:
  Evidence Filter
  Answer Validator
  Abstain when evidence is insufficient
```

These values are the recommended starting baseline for the CropGuard hackathon MVP. They should be evaluated on the actual CropGuard retrieval test set before being treated as production-optimal. The architecture keeps chunking, embedding, retrieval and reranking behind adapters so these values can be tuned without changing the three-agent design.

---

## Final MVP Success Criteria

The MVP is considered ready for the hackathon demonstration when it can successfully:

```text
✓ Load existing Nemotron Omni
✓ Load validated CropGuard LoRA
✓ Process a crop image
✓ Produce diagnosis
✓ Accept farmer question
✓ Maintain conversation context
✓ Incorporate location
✓ Incorporate soil/weather context
✓ Apply Organic/IPM/General preference
✓ Retrieve evidence
✓ Rerank evidence
✓ Generate grounded answer
✓ Validate answer
✓ Abstain when evidence is insufficient
✓ Switch UI dynamically between 10 languages
✓ Generate final response in selected language
✓ Display evidence and grounding status
✓ Persist/update case memory
✓ Run on the target NVIDIA DGX environment
```
