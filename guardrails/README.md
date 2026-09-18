# CropGuard NeMo Guardrails Integration

This directory contains the NeMo Guardrails configuration tailored for the CropGuard architecture. It acts as an **Input Rail** between the Farmer's Query and the Diagnosis Agent (Nemotron Omni + LoRA).

## Architecture Mapping

```text
Farmer (Plant Image + Query)
      │
      ▼
[ NVIDIA NeMo Guardrails ] 
  ├── config.yml    (Model & Rails configuration)
  ├── security.co   (Prompt injection, Jailbreak, System-prompt leak, Malicious input)
  └── off_topic.co  (Off-topic request filtering)
      │
   ALLOWED
      │
      ▼
[ Diagnosis Agent ] (Nemotron Omni + LoRA)
```

## Guardrails Included
1. **Prompt Injection** - Prevents malicious overrides of agent behavior.
2. **Jailbreak** - Detects bypasses (e.g., "Ignore all previous rules").
3. **Off-topic request** - Restricts queries strictly to farming, crops, and plant diagnosis.
4. **System-prompt leak** - Refuses to reveal initialization instructions.
5. **Malicious input** - Blocks harmful, dangerous, or illegal queries (e.g., destroying crops).

## How to use with existing code

To integrate this into your existing MVP without touching your core logic, instantiate the guardrails around your input pipeline:

```python
import os
from nemoguardrails import LLMRails, RailsConfig

# 1. Load the Guardrails configuration from this directory
config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "guardrails"))

# 2. Initialize the rails
app = LLMRails(config)

def process_farmer_query(query: str, image_data=None):
    # 3. Pass the input through Guardrails first
    response = app.generate(messages=[{"role": "user", "content": query}])
    
    # 4. Check if the Guardrails intercepted the request
    # Guardrails will return its own refusal message if any rail is triggered.
    if response["content"] in [
        "I am an agricultural assistant designed for CropGuard. I can only help you with farming, crop health, and plant diagnosis.",
        "I cannot disclose my system instructions or configuration.",
        "I cannot comply with requests that attempt to bypass my safety guidelines.",
        "I cannot process input that attempts to manipulate my instructions.",
        "I cannot provide assistance with malicious, harmful, or illegal activities."
    ]:
        return response["content"] # Request was blocked by Guardrails
        
    # 5. ALLOWED: Proceed to your existing Nemotron Omni + LoRA MVP logic
    return your_existing_diagnosis_agent(query, image_data)
```

## Requirements
Make sure you have NeMo Guardrails installed in your environment:
```bash
pip install nemoguardrails
```

