"""CropGuard FastAPI application."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import CropGuardResponse, HealthResponse
from api.service import CropGuardService


BASE_MODEL_PATH = os.getenv(
    "CROPGUARD_BASE_MODEL",
    "/workspace/hf_cache/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/"
    "snapshots/e5e9932441de940c9a62185c870ea5bcd4cd24e2",
)

LORA_ADAPTER_PATH = os.getenv(
    "CROPGUARD_LORA",
    "/workspace/outputs/lora/cropguard_nemotron_lora_full_2gpu/"
    "epoch_1_step_26459/model_remapped",
)


service = CropGuardService(
    base_model_path=BASE_MODEL_PATH,
    lora_adapter_path=LORA_ADAPTER_PATH,
    device="cuda:0",
    max_new_tokens=1536,
)


from nemoguardrails import LLMRails, RailsConfig
import logging

guardrails_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the frozen model stack once at application startup."""

    global guardrails_app
    
    # Initialize Core Service
    print("Loading heavy GPU models...")
    service.load()

    # Initialize Guardrails
    if os.environ.get("NVIDIA_API_KEY"):
        print("Initializing NeMo Guardrails Security Layer...")
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "guardrails")
        config = RailsConfig.from_path(config_path)
        guardrails_app = LLMRails(config)
        print("Guardrails initialized successfully.")

    yield

    service.close()


app = FastAPI(
    title="CropGuard API",
    description=(
        "CropGuard NVIDIA-powered crop disease diagnosis and "
        "evidence-grounded agricultural advisory API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """Return API readiness."""

    return HealthResponse(**service.health())


@app.post(
    "/api/v1/diagnose",
    response_model=CropGuardResponse,
)
async def diagnose(
    image: UploadFile = File(...),
    farmer_query: str = Form(...),
    preferred_language: str = Form("English"),
    location: str | None = Form(None),
    recommendation_preference: str = Form("General"),
    farmer_id: str | None = Form(None),
    farmer_name: str | None = Form(None),
) -> CropGuardResponse:
    """Run the complete CropGuard pipeline."""

    if not image.content_type:
        raise HTTPException(
            status_code=400,
            detail="Image content type is missing.",
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    suffix = os.path.splitext(
        image.filename or ".jpg"
    )[1] or ".jpg"

    temp_path: str | None = None

    try:
        data = await image.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        # --- NeMo Guardrails Security Check ---
        global guardrails_app
        if guardrails_app is not None:
            try:
                # Run the farmer_query through the LLM Guardrails
                gr_response = await guardrails_app.generate_async(messages=[{"role": "user", "content": farmer_query}])
                gr_content = gr_response["content"].strip()
                
                # Refusal messages defined in our .co flows
                refusals = [
                    "I cannot disclose", "I cannot comply", "I cannot process",
                    "I cannot provide", "Please keep the conversation respectful",
                    "I will not engage", "I am an agricultural assistant",
                    "I'm sorry, I can't respond to that"
                ]
                
                # If the response contains any of the refusal messages, block it instantly!
                if any(r in gr_content for r in refusals):
                    from api.schemas import DiagnosisResponse
                    return CropGuardResponse(
                        status="BLOCKED_BY_GUARDRAILS",
                        answer=gr_content,  # This will be displayed in the UI as the grounded recommendation
                        diagnosis=DiagnosisResponse(crop="N/A", disease="N/A", symptoms="N/A", confidence="BLOCKED"),
                        recommendation_preference=recommendation_preference,
                        language=preferred_language,
                        evidence_count=0,
                        evidence=[],
                        model_called=False
                    )
            except Exception as e:
                # If the Guardrails NVIDIA API fails (e.g. rate limit, 503, network error),
                # log the error but allow the request to pass to the main local PyTorch pipeline
                print(f"[WARNING] Guardrails API check failed, bypassing security: {e}")
        # --------------------------------------


        return service.diagnose(
            image_path=temp_path,
            farmer_query=farmer_query,
            preferred_language=preferred_language,
            location=location,
            recommendation_preference=(
                recommendation_preference
            ),
            farmer_id=farmer_id,
            farmer_name=farmer_name,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CropGuard pipeline failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/")
async def root() -> JSONResponse:
    """Basic API information."""

    return JSONResponse(
        {
            "application": "CropGuard",
            "version": "1.0",
            "status": service.health()["status"],
            "docs": "/docs",
            "health": "/health",
        }
    )

