#!/usr/bin/env bash
# Run from the shared checkout on a network-enabled host, with the venv active.
set -euo pipefail
umask 077
test -f code/score.py
mkdir -p input archives years pilot logs
export HF_HOME="$PWD/.hf-cache"
python code/roster.py
curl --retry 3 -fsSL https://zenodo.org/api/records/15062212 -o input/zenodo_15062212.part
python - <<'PY'
import json
from pathlib import Path
p = Path("input/zenodo_15062212.part")
files = {f["key"] for f in json.loads(p.read_text())["files"]}
assert all(f"g_detail_desc_text_{y}.tsv.zip" in files for y in range(1976, 2025))
p.replace("input/zenodo_15062212.json")
PY
python code/cache_model.py
