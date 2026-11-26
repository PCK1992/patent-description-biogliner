"""Score original description offsets using the Helix keep-all specification."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from importlib.metadata import version

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import inference
from cache_model import MAIN
MODEL = "Ihor/gliner-biomed-bi-large-v1.0"
SPAN = pa.schema([("patent_id", pa.string()), ("description_part", pa.int32()),
                  ("start", pa.int32()), ("end", pa.int32()), ("text", pa.string()),
                  ("label", pa.string()), ("score", pa.float32()), ("source_file", pa.string())])


def load_pinned(device="cuda"):
    from huggingface_hub import snapshot_download
    from gliner import GLiNER
    path = snapshot_download(MODEL, revision=MAIN, local_files_only=True)
    assert Path(path).name == MAIN
    return GLiNER.from_pretrained(path, revision=MAIN).to(device).eval()


def main(year, shard, nshards, pilot):
    labels = json.loads(Path(__file__).with_name("labels.json").read_text())
    assert torch.cuda.is_available() and len(set(labels)) == len(labels) == 127
    assert hashlib.sha256(json.dumps(labels).encode()).hexdigest() == "90c4f611f1dc6856fa1df8aab379544c239ce32319121c59c6984495eef193ec"
    assert version("gliner") == "0.2.28" and version("transformers") == "5.13.1"
    model = load_pinned()
    tokenizer = model.data_processor.transformer_tokenizer
    def count(text):
        return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])
    assert Path(f"years/{year}.json").is_file()
    source = Path(f"years/{year}.parquet")
    pf = pq.ParquetFile(source)
    assert 0 <= shard < nshards
    if pilot:
        groups = sorted(set(int(i * (pf.num_row_groups - 1) / 11) for i in range(12)))
        base = Path("pilot") / str(year)
    else:
        groups = range(shard, pf.num_row_groups, nshards)
        base = Path("scored") / str(year)
    base.mkdir(parents=True, exist_ok=True)
    signature = dict(model=MODEL, revision=MAIN,
                     backend_sha256=hashlib.sha256(Path(inference.__file__).read_bytes()).hexdigest(),
                     source=source.name, source_archive=json.loads(source.with_suffix(".json").read_text())["archive_checksum"],
                     labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
                     runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pilot=pilot)
    def checked(text, max_tokens, overlap, count):
        wins = inference.chunk_text(text, max_tokens, overlap, count)
        finish = 0
        for offset, win in wins:
            if (text[offset:offset+len(win)] != win or text[finish:offset].strip()
                    or count(win) > max_tokens):
                raise ValueError("Window mismatch, omitted text, or over-budget window")
            finish = max(finish, offset+len(win))
        if text[finish:].strip():
            raise ValueError("Uncovered document tail")
        return wins
    for group in groups:
        done = base / f"rg{group:05d}"
        if done.is_dir():
            assert json.loads((done/"manifest.json").read_text())["signature"] == signature
            continue
        temp = done.with_suffix(".partial")
        if temp.exists():
            shutil.rmtree(temp)  # only this locked task's unpublished group
        temp.mkdir()
        rows = pf.read_row_group(group).to_pylist()
        if pilot:
            valid = [r for r in rows if r["text_status"] == "ok"]
            rows = [min(valid, key=lambda r: hashlib.sha256(r["patent_id"].encode()).digest())] if valid else []
        stats = Counter()
        docs = []
        tick = time.monotonic()
        with pq.ParquetWriter(temp/"spans.parquet", SPAN, compression="zstd") as writer:
            for r in rows:
                pid, text, status = r["patent_id"], r["description_text"], r["text_status"]
                n = n50 = None
                if status == "ok":
                    with torch.inference_mode():
                        hits = inference.spans(text, model, labels, stats, checked(text, 1500, 40, count))
                        table = pa.Table.from_pylist([dict(patent_id=pid, description_part=0,
                            source_file=source.name, **hit) for hit in hits], schema=SPAN)
                    n = table.num_rows
                    n50 = 0
                    for s in table.to_pylist():
                        a, b, v = s["start"], s["end"], s["score"]
                        if not (s["patent_id"] == pid and s["label"] in labels and
                                math.isfinite(v) and .1-1e-7 <= v <= 1 and
                                0 <= a < b <= len(text) and text[a:b] == s["text"]):
                            raise ValueError(f"Invalid span: patent={pid}, offsets={a}:{b}")
                        n50 += int(v >= .5)
                    if n:
                        writer.write_table(table)
                docs.append(dict(patent_id=pid, status=status, characters=len(text) if text is not None else None,
                                 spans_floor=n, spans_at_050=n50))
        assert stats["truncation_warnings"] == 0
        assert pq.ParquetFile(temp/"spans.parquet").metadata.num_rows == sum(r["spans_floor"] or 0 for r in docs)
        (temp/"documents.json").write_text(json.dumps(docs)+"\n")
        meta = dict(signature=signature, documents=len(docs), stats=dict(stats),
                    elapsed_seconds=time.monotonic()-tick, gpu=torch.cuda.get_device_name(0),
                    versions={p:version(p) for p in ("torch","gliner","transformers","pyarrow")},
                    model_settings=dict(threshold=.1, nested=True, multi_label=True, window=1500, overlap_words=40))
        (temp/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
        temp.rename(done)
        print(f"{year}/{group}: docs={len(docs)}, spans={stats['spans']}", flush=True)
    if pilot:
        docs = [r for g in groups for r in json.loads((base/f"rg{g:05d}/documents.json").read_text())]
        assert len(docs) >= 6 and sum(r["spans_at_050"] for r in docs)>0
        assert sum(json.loads((base/f"rg{g:05d}/manifest.json").read_text())["stats"].get("extra_windows",0) for g in groups)>0
        (base/"PASS.json").write_text(json.dumps(dict(signature=signature, n=len(docs)))+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if not args.pilot:
        pilot = json.loads(Path("pilot/1976/PASS.json").read_text())
        roster = json.loads(Path("original_roster_audit.json").read_text())
        coverage = json.loads(Path("texts_complete.json").read_text())
        if not (pilot["n"] >= 6 and pilot["signature"]["source"] == "1976.parquet"
                and roster["passed"] is True and coverage["roster"] == roster["original_unique"]
                and coverage["nonempty"] > 0):
            raise RuntimeError("Production requires the 1976 pilot, roster audit and text audit")
    with open(f"logs/score_{args.year}_{args.shard}_{args.nshards}_{args.pilot}.lock","a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.year, args.shard, args.nshards, args.pilot)
