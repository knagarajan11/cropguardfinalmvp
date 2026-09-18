from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv(
    "CROPGUARD_API_URL",
    "http://127.0.0.1:8002",
)


st.set_page_config(
    page_title="CropGuard",
    page_icon="🌱",
    layout="wide",
)


def call_diagnosis_api(
    image_bytes: bytes,
    image_name: str,
    farmer_query: str,
    preferred_language: str,
    recommendation_preference: str,
) -> dict[str, Any]:
    files = {
        "image": (
            image_name,
            image_bytes,
            "image/jpeg",
        )
    }

    data = {
        "farmer_query": farmer_query,
        "preferred_language": preferred_language,
        "recommendation_preference": recommendation_preference,
    }

    response = requests.post(
        f"{API_URL}/api/v1/diagnose",
        files=files,
        data=data,
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        st.info("No evidence was returned.")
        return

    for idx, item in enumerate(evidence, start=1):
        with st.expander(
            f"Evidence {idx}: {item.get('knowledge_id', 'unknown')}"
        ):
            st.write(
                f"**Source:** {item.get('source', 'N/A')}"
            )
            st.write(
                f"**Source type:** {item.get('source_type', 'N/A')}"
            )
            st.write(
                f"**Preference:** "
                f"{item.get('recommendation_preference', 'N/A')}"
            )
            st.write(
                f"**Evidence type:** "
                f"{item.get('evidence_type', 'N/A')}"
            )
            st.write(
                f"**Reranker score:** "
                f"{item.get('reranker_score', 'N/A')}"
            )
            st.write(
                f"**Vector score:** "
                f"{item.get('vector_score', 'N/A')}"
            )
            st.write(
                f"**BM25 score:** "
                f"{item.get('bm25_score', 'N/A')}"
            )
            st.write(
                f"**Hybrid score:** "
                f"{item.get('hybrid_score', 'N/A')}"
            )
            st.write(
                f"**Chunk ID:** `{item.get('chunk_id', 'N/A')}`"
            )


def render_traceability(
    traceability: list[dict[str, Any]],
) -> None:
    if traceability:
        st.json(traceability)
    else:
        st.info("No traceability information returned.")


st.title("🌱 CropGuard")
st.caption(
    "AI-powered crop disease diagnosis and grounded agricultural advisory"
)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Upload Plant Image")

    uploaded_file = st.file_uploader(
        "Upload a leaf or plant image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            width='stretch',
        )

with right:
    st.subheader("2. Farmer Context")

    farmer_query = st.text_area(
        "What would you like to know?",
        value=(
            "How can I prevent this disease? "
        ),
        height=120,
    )

    preferred_language = st.selectbox(
        "Preferred language",
        [
            "English",
            "Hindi",
            "Tamil",
            "Telugu",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Bengali",
            "Gujarati",
            "Punjabi",
        ],
    )

    recommendation_preference = st.selectbox(
        "Recommendation preference",
        ["General", "IPM", "Organic"],
    )

st.divider()

diagnose_clicked = st.button(
    "🔍 Diagnose Crop",
    type="primary",
    width='stretch',
)

if diagnose_clicked:
    if uploaded_file is None:
        st.error("Please upload a plant image first.")
        st.stop()

    if not farmer_query.strip():
        st.error("Please enter a farmer question.")
        st.stop()

    with st.spinner(
        "Analyzing image and retrieving grounded evidence..."
    ):
        try:
            result = call_diagnosis_api(
                image_bytes=uploaded_file.getvalue(),
                image_name=uploaded_file.name,
                farmer_query=farmer_query,
                preferred_language=preferred_language,
                recommendation_preference=recommendation_preference,
            )
        except requests.exceptions.RequestException as exc:
            st.error(
                "CropGuard API is unavailable. "
                f"Details: {exc}"
            )
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    st.divider()
    st.subheader("🌿 CropGuard Diagnosis")

    diagnosis = result.get("diagnosis", {})

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Crop",
            diagnosis.get("crop") or "Unknown",
        )

    with metric2:
        st.metric(
            "Disease",
            diagnosis.get("disease") or "Unknown",
        )

    with metric3:
        st.metric(
            "Confidence",
            diagnosis.get("confidence") or "Unknown",
        )

    symptoms = diagnosis.get("symptoms")
    if symptoms:
        st.info(f"**Visual symptoms:** {symptoms}")

    status = result.get("status", "UNKNOWN")

    if status == "SUFFICIENT":
        st.success("Evidence status: SUFFICIENT")
    elif status == "INSUFFICIENT":
        st.warning(
            "Evidence is insufficient for a grounded recommendation."
        )
    elif status == "CONFLICTING":
        st.warning(
            "Conflicting evidence detected. "
            "CropGuard is avoiding an unsupported recommendation."
        )
    else:
        st.warning(f"Evidence status: {status}")

    st.subheader("💡 Grounded Recommendation")

    answer = result.get("answer", "")
    if answer:
        st.write(answer)
    else:
        st.info(
            "CropGuard could not provide a grounded recommendation."
        )

    st.subheader("📚 Evidence")

    st.caption(
        f"Evidence used: {result.get('evidence_count', 0)}"
    )

    render_evidence(result.get("evidence", []))

    st.subheader("🔗 Traceability")

    render_traceability(
        result.get("traceability", [])
    )

    with st.expander("⚙️ Technical Execution Details"):
        st.write(
            f"**Model called:** "
            f"{result.get('model_called', False)}"
        )
        st.write(
            f"**Generated tokens:** "
            f"{result.get('generated_token_count', 0)}"
        )
        st.write(
            f"**Truncated:** "
            f"{result.get('truncated', False)}"
        )
        st.write(
            f"**Inference time:** "
            f"{result.get('inference_time_seconds', 0):.2f} seconds"
        )
        st.write(
            f"**Language:** "
            f"{result.get('language', 'N/A')}"
        )
        st.write(
            f"**Preference:** "
            f"{result.get('recommendation_preference', 'N/A')}"
        )
