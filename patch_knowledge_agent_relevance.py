from pathlib import Path
import shutil

path = Path("/workspace/agents/knowledge_agent.py")
backup = path.with_suffix(".py.bak2")

shutil.copy2(path, backup)

text = path.read_text()

old = '''        filtered: EvidenceFilterResult = (
            self.evidence_filter.filter(
                reranked_results,
                crop=retrieval_crop,
                disease=retrieval_disease,
                evidence_type=evidence_type,
                recommendation_preference=(
                    recommendation_preference
                ),
            )
        )
'''

new = '''        # ---------------------------------------------------------
        # Preference-specific relevance handling
        #
        # For explicit Organic/IPM requests, candidates have
        # already been restricted to the requested policy before
        # reranking. The NVIDIA reranker remains a ranking signal,
        # while the Evidence Filter enforces policy, context,
        # eligibility, and evidence type.
        #
        # General recommendations retain the configured global
        # relevance threshold.
        # ---------------------------------------------------------

        filter_min_relevance = (
            None
            if recommendation_preference in {"Organic", "IPM"}
            else self.evidence_filter.min_reranker_score
        )

        filtered: EvidenceFilterResult = (
            self.evidence_filter.filter(
                reranked_results,
                crop=retrieval_crop,
                disease=retrieval_disease,
                evidence_type=evidence_type,
                recommendation_preference=(
                    recommendation_preference
                ),
                min_relevance=filter_min_relevance,
            )
        )
'''

if old not in text:
    raise SystemExit(
        "ERROR: Expected EvidenceFilter call was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup:  {backup}")
