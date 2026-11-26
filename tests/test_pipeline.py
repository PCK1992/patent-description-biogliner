import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import hashlib
from collections import Counter
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import inference
from roster import verify


class PipelineTests(unittest.TestCase):
    def test_windows_and_overlapping_label_scores(self):
        text = "aa bb joint cc dd"
        wins = inference.chunk_text(text, 3, 1, lambda s: len(s.split()))
        self.assertEqual(wins, [(0,"aa bb joint"),(6,"joint cc dd")])
        model = Mock()
        def predictions(texts, labels, **kwargs):
            return [[dict(start=t.index("joint"), end=t.index("joint")+5, label=label,
                          score=.7 if t.startswith("joint") else .5, text="joint")
                     for label in labels] for t in texts]
        model.batch_predict_entities.side_effect = predictions
        out = inference.spans(text, model, ["A","B"], Counter(), wins)
        self.assertEqual([(s["start"],s["end"],s["label"],s["score"]) for s in out],
                         [(6,11,"A",.7),(6,11,"B",.7)])
        with self.assertRaises(ValueError):
            inference.chunk_text("toolong",2,1,len)

    def test_cohort_year_and_missing_text(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d)
            for path in ["years","archives","input"]:
                (w/path).mkdir()
            roster=w/"input/roster_for_thresholds.parquet"
            pq.write_table(pa.table({"patent_id":["1","2","3"],"grant_year":[1976]*3}),roster)
            with self.assertRaises(ValueError):
                verify(roster)
            name="g_detail_desc_text_1976.tsv.zip"
            archive=w/"archives"/name
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr(name[:-4],"patent_id\tdescription_text\tdescription_length\n1\tA device\t8\n2\t\t0\n4\tother\t5\n")
            meta={"key":name,"size":archive.stat().st_size,"checksum":"md5:"+hashlib.md5(archive.read_bytes()).hexdigest()}
            (w/"input/zenodo_15062212.json").write_text(json.dumps({"files":[meta]}))
            for _ in range(2):
                subprocess.run([sys.executable,ROOT/"code/build_year.py","1976"],cwd=w,check=True,capture_output=True)
            t=pq.read_table(w/"years/1976.parquet").to_pylist()
            self.assertEqual([(r["patent_id"],r["text_status"]) for r in t],
                             [("1","ok"),("2","empty"),("3","absent")])

    def test_all_submits_full_array_with_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d)
            (w/"code").symlink_to(ROOT/"code")
            (w/"input").mkdir()
            for path in ["input/zenodo_15062212.json","original_roster_audit.json"]:
                (w/path).write_text("{}")
            stub=w/"sbatch"
            stub.write_text('#!/bin/sh\necho "$*" >> calls\nwc -l < calls | tr -d " "\n')
            stub.chmod(0o700)
            py=w/"python"
            py.symlink_to(sys.executable)
            env={**os.environ,"PATH":d+":"+os.environ["PATH"],"MAX_GPUS":"8"}
            env.pop("SHARDS", None)
            subprocess.run(["bash",ROOT/"scripts/submit.sh","all"],cwd=w,env=env,check=True,capture_output=True)
            calls=(w/"calls").read_text().splitlines()
            self.assertEqual(len(calls),5)
            self.assertIn("afterok:1_0",calls[2])
            self.assertIn("--array=0-391%8",calls[3])
            self.assertIn("afterok:2:3",calls[3])
            self.assertIn("afterok:4",calls[4])
