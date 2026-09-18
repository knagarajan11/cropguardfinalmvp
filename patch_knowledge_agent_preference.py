from pathlib import Path
import shutil

path = Path("/workspace/agents/knowledge_agent.py")
backup = path.with_suffix(".py.bak")

shutil.copy2(path, backup)

text = path.read_text()

old = """        retrieval_results = self.retriever.search(
            query=query,
            top_k=self.candidate_k,
            crop=retrieval_crop,
            disease=retrieval_disease,
        )

        # ---------------------------------------------------------
        # 2. NVIDIA Reranker
        # ---------------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            candidates=retrieval_results,
            top_k=requested_top_k,
        )
"""

new = """        retrieval_results = self.retriever.search(
            query=query,
            top_k=self.candidate_k,
            crop=retrieval_crop,
            disease=retrieval_disease,
        )

        # ---------------------------------------------------------
        # Preference-aware candidate selection
        #
        # For explicit Organic/IPM requests, matching-policy
        # evidence must not be crowded out by General evidence
        # before reranking.
        #
        # The Evidence Filter remains the final authoritative
        # policy and safety gate.
        #
        # Original KB metadata is preserved unchanged.
        # ---------------------------------------------------------

        rerank_candidates = retrieval_results

        if recommendation_preference in {"Organic", "IPM"}:
            matching_candidates = []

            requested_preference = (
                recommendation_preference.strip().lower()
            )

            for item in retrieval_results:
                metadata = getattr(item, "metadata", None)

                if metadata is None and isinstance(item, dict):
                    metadata = item.get("metadata", {})

                metadata = dict(metadata or {})

                actual_preference = str(
                    metadata.get(
                        "recommendation_preference",
                        "",
                    )
                ).strip().lower()

                if actual_preference == requested_preference:
                    matching_candidates.append(item)

            # If explicit policy evidence exists, rerank only
            # that policy-specific candidate set.
            #
            # If none exists, preserve the original candidates so
            # the Evidence Filter can safely abstain.
            if matching_candidates:
                rerank_candidates = matching_candidates

        # ---------------------------------------------------------
        # 2. NVIDIA Reranker
        # ---------------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            candidates=rerank_candidates,
            top_k=requested_top_k,
        )
"""

if old not in text:
    raise SystemExit(
        "ERROR: Expected retrieval/reranker block was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup:  {backup}")
