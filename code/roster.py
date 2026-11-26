"""Verify the exact original cohort; optionally export only IDs and grant years."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

KEYS = ["patent_id", "grant_year"]
EXPECTED = "d29207dc188996acda8f85fd85ec1141c00beb98e006a597c99bd55bb78abda0"
N = 1_403_822


def verify(source):
    table = pq.read_table(source, columns=KEYS).sort_by("patent_id").replace_schema_metadata(None)
    digest = hashlib.sha256()
    for batch in table.to_batches(max_chunksize=65536):
        digest.update("".join(f"{p}\t{y}\n" for p, y in zip(*[c.to_pylist() for c in batch.columns])).encode())
    if table.num_rows != N or digest.hexdigest() != EXPECTED:
        raise ValueError("Patent IDs or years differ from the verified original roster.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, nargs="?", default=Path("input/roster_for_thresholds.parquet"))
    ap.add_argument("--export", type=Path, help="Write the two-column collaborator input outside Git")
    args = ap.parse_args()
    t = verify(args.source)
    if args.export:
        if args.export.exists():
            raise FileExistsError(args.export)
        args.export.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, args.export, compression="zstd")
        verify(args.export)
    else:
        Path("original_roster_audit.json").write_text(json.dumps(dict(
            passed=True, original_unique=N, key_sha256=EXPECTED,
            method="ID/year digest verified against the previously reconciled original C15 roster"
        ), indent=2) + "\n")
    print(f"Verified {N:,} patent IDs and grant years.")
