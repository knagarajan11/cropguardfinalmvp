from pathlib import Path
import shutil

path = Path("/workspace/evidence/evidence_filter.py")
backup = path.with_suffix(".py.bak2")

shutil.copy2(path, backup)

text = path.read_text()

old = '''    def _relevance_check(
        self,
        item: Any,
    ) -> tuple[bool, str | None]:

        if not self.require_relevant_evidence:
            return True, None

        score = self._score(item)

        if score is None:
            return (
                False,
                "Missing or invalid reranker relevance score.",
            )

        if self.min_reranker_score is not None:
            if score < self.min_reranker_score:
                return (
                    False,
                    f"Weak relevance: score {score:.6f} "
                    f"is below configured threshold "
                    f"{self.min_reranker_score:.6f}.",
                )

        return True, None
'''

new = '''    def _relevance_check(
        self,
        item: Any,
        threshold: float | None = None,
    ) -> tuple[bool, str | None]:

        if not self.require_relevant_evidence:
            return True, None

        score = self._score(item)

        if score is None:
            return (
                False,
                "Missing or invalid reranker relevance score.",
            )

        if threshold is not None:
            if score < threshold:
                return (
                    False,
                    f"Weak relevance: score {score:.6f} "
                    f"is below configured threshold "
                    f"{threshold:.6f}.",
                )

        return True, None
'''

if old not in text:
    raise SystemExit(
        "ERROR: Expected _relevance_check block was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup: {backup}")
