"""Build one C15 year on Helix, preserving missing descriptions and verbatim offsets."""
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq


def build(year):
    manifest = json.loads(Path("input/zenodo_15062212.json").read_text())
    name = f"g_detail_desc_text_{year}.tsv.zip"
    source, = [r for r in manifest["files"] if r["key"] == name]
    roster = pq.read_table("input/roster_for_thresholds.parquet", columns=["patent_id"],
                           filters=[("grant_year", "=", year)]).column(0).to_pylist()
    wanted = set(map(str, roster))
    assert len(wanted) == len(roster) > 0 and None not in roster
    dest = Path(f"years/{year}.parquet")
    audit = dest.with_suffix(".json")
    if audit.exists():
        assert pq.ParquetFile(dest).metadata.num_rows == len(wanted)
        return
    arc = Path("archives") / name
    if not arc.exists():
        part = arc.with_suffix(".zip.part")
        if not part.exists() or part.stat().st_size != source["size"]:
            subprocess.run(["curl", "-fsSL", "--retry", "4", "--connect-timeout", "30",
                            "--max-time", "10800", "-C", "-", "-o", str(part),
                            f"https://zenodo.org/records/15062212/files/{name}?download=1"], check=True)
    candidate = arc if arc.exists() else part
    assert candidate.stat().st_size == source["size"], name
    with candidate.open("rb") as stream:
        assert "md5:" + hashlib.file_digest(stream, "md5").hexdigest() == source["checksum"], name
    if candidate != arc:
        candidate.replace(arc)
    seen = set()
    nonempty = empty = national = 0
    batch = []
    schema = pa.schema([("patent_id", pa.string()), ("description_text", pa.large_string()),
                        ("text_status", pa.string())])
    tmp = dest.with_suffix(".partial.parquet")
    csv.field_size_limit(sys.maxsize)
    with pq.ParquetWriter(tmp, schema, compression="zstd", use_dictionary=["text_status"]) as out:
        def emit(pid, text, status):
            batch.append((pid, text, status))
            if len(batch) >= 64:
                flush()

        def flush():
            if batch:
                out.write_table(pa.Table.from_arrays(list(zip(*batch)), schema=schema))
                batch.clear()

        with zipfile.ZipFile(arc) as z, z.open(name[:-4]) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""),
                                delimiter="\t", strict=True)
            assert next(reader) == ["patent_id", "description_text", "description_length"]
            for pid, text, length in reader:
                national += 1
                if pid not in wanted:
                    continue
                if pid in seen:
                    raise ValueError(f"Duplicate C15 description: {year} {pid}")
                seen.add(pid)
                if text.strip():
                    emit(pid, text, "ok")
                    nonempty += 1
                else:
                    emit(pid, None, "empty")
                    empty += 1
        for pid in sorted(wanted - seen):
            emit(pid, None, "absent")
        flush()
    got = pq.read_table(tmp, columns=["patent_id"]).column(0).to_pylist()
    assert len(got) == len(wanted) and set(got) == wanted and nonempty > 0
    tmp.replace(dest)
    result = dict(year=year, roster=len(roster), nonempty=nonempty, empty=empty,
                  absent=len(wanted-seen), national_rows=national, archive=name,
                  archive_checksum=source["checksum"], source="PatentsView detailed description only")
    temp = audit.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(audit)
    print(result, flush=True)


if __name__ == "__main__":
    year = int(sys.argv[1])
    assert 1976 <= year <= 2024
    with open(f"years/{year}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build(year)
