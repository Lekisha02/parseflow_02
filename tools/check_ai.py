"""Check that the AI model configured for ParseFlow answers: one short real request (and an image check).

    python tools/check_ai.py          # reads .env in the project folder, like ./run.sh

Prints what happened in plain words: the provider and model, whether it answered, how long it took, and whether it
read a test image. Your key is never printed.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        v = re.sub(r"\s+#.*$", "", v).strip()
        if v:
            os.environ.setdefault(k.strip(), v)
sys.path.insert(0, str(ROOT / "backend"))
from processing import llm  # noqa: E402

r = llm.selftest()
print(json.dumps(r, indent=1))
if r.get("ok"):
    print(f"\nOK: {r['provider']} / {r['model']} answered in {r['seconds']} s." +
          (f" Image test: {'passed' if r.get('vision_ok') else 'failed ' + str(r.get('vision_error') or r.get('vision_reply'))}." if r.get("vision") else ""))
    sys.exit(0)
print(f"\nNOT WORKING: {r.get('error')}")
print("Check ASK_PROVIDER, ASK_API_KEY and ASK_MODEL in .env (for Gemini: ASK_PROVIDER=gemini, a key from aistudio.google.com).")
sys.exit(1)
