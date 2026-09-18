from pathlib import Path
import shutil

path = Path("/workspace/evidence/evidence_filter.py")
backup = path.with_suffix(".py.bak3")

shutil.copy2(path, backup)

text = path.read_text()

old = '''            else:
                relevance_ok, reason = self._relevance_check(item)

                if not relevance_ok:
'''

new = '''            else:
                relevance_ok, reason = self._relevance_check(
                    item,
                    threshold=threshold,
                )

                if not relevance_ok:
'''

if old not in text:
    raise SystemExit(
        "ERROR: Expected _relevance_check call was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup: {backup}")
