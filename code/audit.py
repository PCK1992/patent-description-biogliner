"""Reconcile complete text and NER ledgers. Never count an unscored document as zero."""
import json
import os
from pathlib import Path
import sys
from collections import Counter
import pyarrow.parquet as pq
import pyarrow.compute as pc
from production import text_receipt
from roster import EXPECTED, verify
from score import OLD_HASHES, signature_for
from window_repair import compatible

mode = sys.argv[1]
assert mode in {"texts", "scores"}
first, last = (map(int, sys.argv[2].split("-")) if len(sys.argv) == 3 else (0, 391))
assert 0 <= first <= last <= 391
start, end = 1976+first//8, 1976+last//8
partial_years = (start,end) != (1976,2024)
if partial_years:
    roster = verify("input/roster_for_thresholds.parquet")
totals = Counter()
for year in range(start, end+1):
    meta = json.loads(Path(f"years/{year}.json").read_text())
    src = pq.ParquetFile(f"years/{year}.parquet")
    assert meta["roster"] == src.metadata.num_rows == sum(meta[k] for k in ("nonempty","empty","absent"))
    if partial_years:
        expected = roster.filter(pc.equal(roster["grant_year"],year))["patent_id"].to_pylist()
        assert sorted(map(str,expected)) == sorted(src.read(columns=["patent_id"])["patent_id"].to_pylist())
    totals.update({k:meta[k] for k in ("roster","nonempty","absent","empty")})
    if mode == "scores":
        expected_sig = signature_for(f"{year}.parquet", meta["archive_checksum"], False)
        for i in range(src.num_row_groups):
            if not first <= (year - 1976) * 8 + i % 8 <= last:
                continue
            group = Path(f"scored/{year}/rg{i:05d}")
            m = json.loads((group/"manifest.json").read_text())
            sig = m["signature"]
            assert compatible(sig, expected_sig, OLD_HASHES)
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
if not partial_years:
    assert totals["roster"] == 1403822
    assert (totals["nonempty"], totals["absent"], totals["empty"]) == (1402145, 1676, 1)
if mode == "scores" and first%8 == 0 and last%8 == 7:
    assert totals["validated_documents"] == totals["roster"]
if mode == "texts":
    out = text_receipt(start,end)
elif (first,last) == (0,391):
    out = Path("scores_complete.json")
else:
    out = Path(f"scores_tasks_{first}_{last}.json")
receipt = dict(totals,grant_years=[start,end],roster_sha256=EXPECTED)
tmp=out.with_suffix(f".{os.getpid()}.tmp")
tmp.write_text(json.dumps(receipt, indent=2)+"\n")
tmp.replace(out)
print(dict(totals))
