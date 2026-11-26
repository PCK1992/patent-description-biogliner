"""Reconcile complete text and NER ledgers. Never count an unscored document as zero."""
import json
import hashlib
from pathlib import Path
import sys
from collections import Counter
import pyarrow.parquet as pq

mode = sys.argv[1]
assert mode in {"texts", "scores"}
first, last = (map(int, sys.argv[2].split("-")) if len(sys.argv) == 3 else (0, 391))
assert 0 <= first <= last <= 391
assert mode == "scores" or (first, last) == (0, 391)
runner = hashlib.sha256(Path(__file__).with_name("score.py").read_bytes()).hexdigest()
backend = hashlib.sha256(Path(__file__).with_name("inference.py").read_bytes()).hexdigest()
totals = Counter()
for year in range(1976, 2025):
    meta = json.loads(Path(f"years/{year}.json").read_text())
    src = pq.ParquetFile(f"years/{year}.parquet")
    assert meta["roster"] == src.metadata.num_rows == sum(meta[k] for k in ("nonempty","empty","absent"))
    totals.update({k:meta[k] for k in ("roster","nonempty","absent","empty")})
    if mode == "scores":
        for i in range(src.num_row_groups):
            if not first <= (year - 1976) * 8 + i % 8 <= last:
                continue
            group = Path(f"scored/{year}/rg{i:05d}")
            m = json.loads((group/"manifest.json").read_text())
            sig = m["signature"]
            assert not sig["pilot"] and sig["runner_sha256"] == runner
            assert sig["backend_sha256"] == backend
            assert sig["source"] == f"{year}.parquet" and sig["source_archive"] == meta["archive_checksum"]
            assert sig["revision"] == "75fb10d6d5500c5e98c285493578116540ec47d3"
            docs = json.loads((group/"documents.json").read_text())
            ids = src.read_row_group(i, columns=["patent_id"]).column(0).to_pylist()
            assert ids == [r["patent_id"] for r in docs] and m["documents"] == len(ids)
            assert m["stats"].get("truncation_warnings",0) == 0
            n = sum(r["spans_floor"] or 0 for r in docs)
            assert pq.ParquetFile(group/"spans.parquet").metadata.num_rows == n
            assert all((r["spans_floor"] is None) == (r["status"] != "ok") for r in docs)
            totals["spans_floor"] += n
            totals["validated_documents"] += len(ids)
            totals["spans_at_050"] += sum(r["spans_at_050"] or 0 for r in docs)
assert totals["roster"] == 1403822
assert (totals["nonempty"], totals["absent"], totals["empty"]) == (1402145, 1676, 1)
if mode == "texts" or (first, last) == (0, 391):
    if mode == "scores":
        assert totals["validated_documents"] == totals["roster"]
    out = Path(f"{mode}_complete.json")
else:
    out = Path(f"scores_tasks_{first}_{last}.json")
out.write_text(json.dumps(totals, indent=2)+"\n")
print(dict(totals))
