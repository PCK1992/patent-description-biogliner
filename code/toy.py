"""Run the production BioGLiNER inference functions on two synthetic device descriptions."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import torch

from inference import chunk_text, spans
from score import load_pinned


def main(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Run inside a GPU allocation; --cpu is a separate CPU test.")
    labels = json.loads(Path(__file__).with_name("labels.json").read_text())
    model = load_pinned(device)
    tokenizer = model.data_processor.transformer_tokenizer
    def count(s):
        return len(tokenizer(s, add_special_tokens=False, verbose=False)["input_ids"])

    short = "A standing wheelchair uses an accelerometer to detect tilt. A controller adjusts the seat angle."
    # Force a boundary at the actual 1,500-token production limit, not a toy window size.
    long = (short + " ") * 128 + "A pressure sensor measures the load on the footrest."
    name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Device: {name}; torch {torch.__version__}; gliner {version('gliner')}; transformers {version('transformers')}")
    for text, should_split in [(short, False), (long, True)]:
        windows = chunk_text(text, 1500, 40, count)
        stats = Counter()
        with torch.inference_mode():
            hits = spans(text, model, labels, stats, windows)
        if ((len(windows) > 1) != should_split or not hits or stats["truncation_warnings"]
                or any(not (0 <= h["start"] < h["end"] <= len(text)
                            and text[h["start"]:h["end"]] == h["text"]
                            and .1 - 1e-7 <= h["score"] <= 1 and h["label"] in labels) for h in hits)):
            raise ValueError(f"Toy check failed: windows={len(windows)}, expected_split={should_split}, stats={stats}")
        print(f"{len(windows)} window(s), {len(hits)} span-label rows. Example: {max(hits, key=lambda h: h['score'])}")
    print(f"TOY PASS ({device}): 2 synthetic descriptions. No corpus jobs submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Test on CPU explicitly; default requires CUDA.")
    main("cpu" if parser.parse_args().cpu else "cuda")
