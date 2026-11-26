# Score all C15 patent descriptions with BioGLiNER

Run named-entity extraction on the detailed descriptions of **all 1,403,822 patents** in the original biomedical C15 roster (grant years **1976-2024**). The source release contains usable text for **1,402,145** patents; preserve the **1,676 absent** and **one blank** record. Do not restrict to a regression sample, years after burn-in, particular CPCs, or patents with an event.

The [C15 roster](input/roster_for_thresholds.parquet) ships with the repository. It contains only patent IDs and grant years. Setup reads it directly from the checkout; no separate attachment or private account is required. Grant texts come from the [PatentsView 2024 release](https://zenodo.org/records/15062212); weights come from the [BioGLiNER biomedical bi-encoder](https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0).

## Stanford handoff: 2016 onward

After [setup](#1-prepare-a-shared-workspace), run this from the checkout with the virtual environment active. If you already have a checkout, first run `git pull --ff-only`. Set your site's partitions/account if required.

```bash
START_YEAR=2016 MAX_GPUS=10 bash scripts/submit.sh all
```

This scores **568,976 C15 grants from 2016 through 2024** (568,973 nonempty descriptions), in **72 production shards, numbered 320–391**. It prepares only these nine grant years plus 1976 for the short CUDA/offset check. If staging ZIPs manually, stage 1976 and 2016–2024 only. There is no GPU speed benchmark in the submission.

Slurm requests **one GPU and four CPUs per production job**, up to ten simultaneous jobs. Within each job, the launcher starts **three workers per full A100 or H200**, **two per A40**, or one on other GPUs. Each worker scores different row groups; the checkpoint, offsets, labels and confidence scores use the same scorer as before. Ten A100/H200 GPUs can therefore run up to 30 workers, subject to allocation. This is not a 30-fold speedup.

**Helix timing:** On the same 12 descriptions, an A100 40 GB took 33.5 s with two workers/two CPUs, 33.6 s with two workers/four CPUs, and 28.6 s with three workers/four CPUs. The repeat gave 33.3 s vs 28.8 s (16% higher throughput); all saved spans and scores matched. Four workers were slower. These are Helix measurements, not Stanford timings or a measured H200 gain.

Please send the Git commit, Slurm job IDs and the first completed shard number to coordinate the year split before both clusters score the same years. At completion share `texts_years_2016_2024.json`, `scores_tasks_320_391.json` and the output location. Keep each site's original manifests; exchange data through research storage, not GitHub.

## GPU toy example

Start here to test the scorer without downloading any patents. Clone the repository and install the environment as in step 1 below, stopping **before** `bash scripts/prepare.sh`. In the checkout, with that environment active:

```bash
# On a node with internet access: cache the inference weights, not the patent archives.
export HF_HOME="$PWD/.hf-cache"
python code/cache_model.py

# Slurm: add your local account/partition if required. No production jobs are submitted.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:10:00 .venv/bin/python code/toy.py
```

Already inside a GPU allocation? Run `python code/toy.py` directly with the same cache settings. The [toy script](code/toy.py) uses the production checkpoint, labels, thresholds and inference functions on two synthetic wheelchair descriptions: one short and one long enough to require overlapping windows. It prints the device, package versions, example spans and **`TOY PASS (cuda)`** on success. No Stanford GPU is needed to inspect the code, but a CPU pass is not a CUDA pass. For a real-patent pilot only, complete step 1 and run `bash scripts/submit.sh pilot`; this prepares 1976, not all 49 years.

**Files for the full run:** [Download the exact 1.4M-ID C15 roster (Parquet, 3 MB)](https://raw.githubusercontent.com/PCK1992/patent-description-biogliner/38b3ea7/input/roster_for_thresholds.parquet). [Download source descriptions by grant year (Zenodo)](https://zenodo.org/records/15062212). The files named `g_detail_desc_text_YEAR.tsv.zip` contain **all US grants** for that year, not just C15. The pipeline selects the roster IDs and writes the C15-only files to `years/YEAR.parquet`. No manual download is necessary in the usual workflow.

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
3. **Production:** after the pilot and text audit pass, run eight disjoint shards per year. Each job requests one GPU, four CPUs and 32 GB of host RAM. Workers share the GPU and load their own copy of the weights once, offline.
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
