"""Stage only inference files in the shared HF cache before offline Slurm jobs."""
from pathlib import Path
from huggingface_hub import snapshot_download

MAIN = "75fb10d6d5500c5e98c285493578116540ec47d3"
LABELS = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
if __name__ == "__main__":
    snapshot_download("Ihor/gliner-biomed-bi-large-v1.0", revision=MAIN,
                      allow_patterns=["*.json", "spm.model", "pytorch_model.bin"])
    # GLiNER looks up this tokenizer by name, not revision, even when the model is local.
    path = Path(snapshot_download("BAAI/bge-base-en-v1.5", revision=LABELS,
                                  allow_patterns=["*.json", "vocab.txt"]))
    assert path.name == LABELS
    ref = path.parents[1] / "refs" / "main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(LABELS)  # The README specifies a project-private HF_HOME.
    print("Cached both snapshots. GPU workers can run offline.")
