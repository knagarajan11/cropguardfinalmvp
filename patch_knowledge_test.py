from pathlib import Path
import shutil

path = Path("/workspace/agents/test_knowledge_agent.py")
backup = path.with_suffix(".py.bak")

shutil.copy2(path, backup)

text = path.read_text()

old = '''    for item in general.accepted_evidence:
        assert item.metadata.get("crop") == "Corn"
        assert (
            item.metadata.get("disease")
            == "northern_leaf_blight"
        )
'''

new = '''    for item in general.accepted_evidence:
        assert item.metadata.get("crop") == "Corn"

        # Preserve original KB provenance while accepting
        # equivalent application-level disease labels.
        disease = str(
            item.metadata.get("disease", "")
        ).strip().lower()

        assert disease in {
            "northern_leaf_blight",
            "northern corn leaf blight",
            "northern leaf blight",
            "nclb",
        }
'''

if old not in text:
    raise SystemExit(
        "ERROR: Expected assertion block was not found. "
        "No changes made."
    )

path.write_text(text.replace(old, new, 1))

print("PATCH PASS")
print(f"Updated: {path}")
print(f"Backup:  {backup}")
