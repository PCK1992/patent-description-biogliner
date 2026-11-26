import json
import os
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import hashlib
from collections import Counter
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import inference
from production import score_shard, workers_for
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
            for name in ("SHARDS","START_YEAR"):
                env.pop(name, None)
            subprocess.run(["bash",ROOT/"scripts/submit.sh","all"],cwd=w,env=env,check=True,capture_output=True)
            calls=(w/"calls").read_text().splitlines()
            self.assertEqual(len(calls),5)
            self.assertIn("afterok:1_0",calls[2])
            self.assertIn("--array=0-391%8",calls[3])
            self.assertIn("afterok:2:3",calls[3])
            self.assertIn("afterok:4",calls[4])
            (w/"calls").unlink()
            env.update(START_YEAR="2016",MAX_GPUS="10")
            subprocess.run(["bash",ROOT/"scripts/submit.sh","all"],cwd=w,env=env,check=True,capture_output=True)
            years,texts,pilot,score,audit=(w/"calls").read_text().splitlines()
            self.assertIn("--array=0,40-48%3",years)
            self.assertIn("audit texts 320-391",texts)
            self.assertIn("afterok:1_0",pilot)
            for flag in ("--array=320-391%10","--cpus-per-task=4","afterok:2:3","full 320-391"):
                self.assertIn(flag,score)
            self.assertIn("audit scores 320-391",audit)

    def test_workers_partition_and_resume(self):
        self.assertEqual([workers_for(g,4) for g in ("NVIDIA A100-SXM4-40GB","NVIDIA H200","NVIDIA A40","other")],[3,3,2,1])
        old=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                Path("logs").mkdir()
                worker=Path(d)/"fake.py"
                worker.write_text("import os,sys\nfrom pathlib import Path\n"
                                  "fd,yr,pos,mod=map(int,sys.argv[2:])\nos.fstat(fd)\n"
                                  "if pos==8 and not Path('failed_once').exists():\n"
                                  "    Path('failed_once').touch();sys.exit(1)\n"
                                  "for g in range(pos,109,mod): Path(f'group-{g}').touch(exist_ok=False)\n")
                score_shard(2016,0,3,worker)
                self.assertTrue(Path("failed_once").is_file())
                self.assertEqual(sorted(int(p.name[6:]) for p in Path(".").glob("group-*")),list(range(0,109,8)))
            finally:
                os.chdir(old)

    def test_partial_receipts_do_not_certify_full_run(self):
        old=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                Path("years").mkdir()
                pq.write_table(pa.table({"patent_id":["42"]}),"years/2016.parquet")
                Path("years/2016.json").write_text(json.dumps(dict(roster=1,nonempty=1,empty=0,absent=0,archive_checksum="md5:fixture")))
                group=Path("scored/2016/rg00000")
                group.mkdir(parents=True)
                sig={k:hashlib.sha256((ROOT/"code"/p).read_bytes()).hexdigest()
                     for k,p in (("runner_sha256","score.py"),("backend_sha256","inference.py"))}
                sig.update(pilot=False,source="2016.parquet",source_archive="md5:fixture",revision="75fb10d6d5500c5e98c285493578116540ec47d3")
                (group/"manifest.json").write_text(json.dumps(dict(signature=sig,documents=1,stats={})))
                (group/"documents.json").write_text(json.dumps([dict(patent_id="42",status="ok",spans_floor=0,spans_at_050=0)]))
                pq.write_table(pa.table({"score":pa.array([],type=pa.float32())}),group/"spans.parquet")
                with patch("roster.verify",return_value=pa.table({"patent_id":["42"],"grant_year":[2016]})):
                    for mode in ("texts","scores"):
                        with patch.object(sys,"argv",["audit.py",mode,"320-327"]):
                            runpy.run_path(str(ROOT/"code/audit.py"),run_name="__main__")
                        self.assertFalse(Path(f"{mode}_complete.json").exists())
                self.assertTrue(Path("texts_years_2016_2016.json").exists())
                self.assertEqual(json.loads(Path("scores_tasks_320_327.json").read_text())["validated_documents"],1)
            finally:
                os.chdir(old)
