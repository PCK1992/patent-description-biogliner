# Score all C15 patent descriptions with BioGLiNER

Run named-entity extraction on the detailed descriptions of **all 1,403,822 patents** in the original biomedical C15 roster (grant years **1976-2024**). The source release contains usable text for **1,402,145** patents; preserve the **1,676 absent** and **one blank** record. Do not restrict to a regression sample, years after burn-in, particular CPCs, or patents with an event.

The [C15 roster](input/roster_for_thresholds.parquet) ships with the repository. It contains only patent IDs and grant years. Setup reads it directly from the checkout; no separate attachment or private account is required. Grant texts come from the [PatentsView 2024 release](https://zenodo.org/records/15062212); weights come from the [BioGLiNER biomedical bi-encoder](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0).

## 1. Prepare a shared workspace

Use a Linux cluster with Slurm, CUDA-capable GPUs, and persistent shared storage visible from all nodes. Clone **on the cluster**, not on a laptop. Use a private working directory for the generated data. The 49 national description archives alone total **71.2 GB** compressed; allow substantial extra room for annual extracts, weights, and scored spans. Check the actual workspace quota as the first years finish.

```bash
umask 077
git clone https://github.com/PCK1992/patent-description-biogliner.git
cd patent-description-biogliner

# Load a cluster Python (3.13.1 on Helix) and a compatible NVIDIA driver.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests
bash scripts/prepare.sh
```

Prepare on an internet-enabled node. It verifies the cohort against the original ID/year digest, retrieves the public release inventory, and caches both model and label-tokenizer snapshots under `.hf-cache/`. The CPU year-build jobs download and checksum the archives lazily. **If compute nodes cannot reach Zenodo**, stage the ZIPs first on a permitted transfer node:

```bash
seq 1976 2024 | xargs -P 2 -I '{}' curl -fLsS --retry 4 -C - \
  -o 'archives/g_detail_desc_text_{}.tsv.zip' \
  'https://zenodo.org/records/15062212/files/g_detail_desc_text_{}.tsv.zip?download=1'
```

## 2. Submit the full run

From the checkout, with the same venv active:

```bash
# Set these to your site's actual values, or omit those supplied by site defaults.
# export ACCOUNT=... CPU_PARTITION=... GPU_PARTITION=... MAIL_TO=...
# export GRES=gpu:1 GPU_TIME=24:00:00
MAX_GPUS=8 bash scripts/submit.sh all
```

This submits **all 392 shards**, not eight shards. `MAX_GPUS` limits simultaneous production jobs; choose a value approved for your allocation. The dependency chain is:

1. **CPU:** build 49 annual tables; keep original text and explicit missing statuses.
2. **One short GPU pilot:** as soon as 1976 is ready, sample spread-out row groups. Check span offsets, nonzero detections, multi-window handling and truncation. The time limit is 15 minutes. No separate 2019 test.
3. **Production:** after the pilot and full text audit pass, run eight disjoint shards per year. Each worker uses one GPU, two CPUs, and 32 GB of host RAM. It loads the weights once, offline.
4. **Final CPU audit:** validate every patent ID and row group against the source; check model signatures, span counts, text coverage and missingness. The final receipt is `scores_complete.json`.

The jobs survive logout. Record the job IDs printed on submission. No Stanford-specific account, partition, email or GPU type is hardcoded. `MAIL_TO` enables Slurm BEGIN/END/FAIL messages; actual delivery depends on the cluster mail service.

## Outputs and recovery

- `scored/YEAR/rgNNNNN/spans.parquet`: patent ID, description part, start, end, original text span, semantic label, score, source file. Offsets are zero-based Python character positions; the end is exclusive.
- `documents.json` and `manifest.json` beside each shard's output: document status, counts, GPU, software, settings, timing and provenance. A missing description has a null count, not zero.
- `original_roster_audit.json`, `texts_complete.json`, `pilot/1976/PASS.json`, and `scores_complete.json`: send these files and the Git commit to Patrick once the full run passes. Share the full `scored/` directory via approved research storage, **not GitHub**.
- If a job times out, fix the cause and rerun with the same checkout and workspace. For example, `SHARDS=344-344 GPU_TIME=36:00:00 bash scripts/submit.sh production` reruns one task after a successful pilot. Committed row groups are checked and skipped. A restricted run issues only a partial receipt; after all missing tasks finish, run `python code/audit.py scores` on a CPU node. Do not copy partial directories or combine checkpoints from different implementations.

## Measurement and verification

Only the **detailed description** is scored, not claims, abstracts or titles. The public runner reuses the original Helix cohort builder, scoring loop, checkpoint format and audits, with a portable adapter to the public GLiNER API. It implements Miriam's keep-all specification: all 127 ordered UMLS prompts, nested and multi-label spans, score floor **0.10**, **1,500 subword tokens** per window and **40 words** of overlap. Repeated positions stay distinct; identical positions with the same label across windows keep the highest score. A later 0.50 analysis does not require re-encoding.

On a synthetic device description, the public adapter and the pinned reference extractor returned **identical span offsets, labels and scores** using the actual checkpoint on CPU: 178 span-label rows in a single window, 185 with a deliberately smaller budget forcing four overlapping windows. This tests agreement *between implementations at each budget*, not equality between window sizes. The comparison used the available local GLiNER 0.2.22 environment; production pins Helix's 0.2.28. It does not establish full-corpus parity or replace the Stanford GPU pilot.
