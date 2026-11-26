"""Window a single description and reconcile the public GLiNER batched API's spans."""
from bisect import bisect_right
import gc
import re
import warnings

import torch


def chunk_text(text, limit, overlap, count):
    if count(text) <= limit:
        return [(0, text)]
    words = list(re.finditer(r"\S+", text))
    result = []
    pos = 0
    while pos < len(words):
        start = words[pos].start()
        ends = range(pos, len(words))
        n = bisect_right(ends, limit, key=lambda j: count(text[start:words[j].end()]))
        if not n:
            raise ValueError("Single word exceeds the tokenizer budget")
        stop = pos + n
        result.append((start, text[start:words[stop-1].end()]))
        if stop == len(words):
            break
        pos = max(pos + 1, stop - overlap)
    return result


def groups(windows):
    batch = []
    for item in sorted(windows, key=lambda w: len(w[1].split())):
        if batch and (len(batch) == 16 or (len(batch)+1)*len(item[1].split()) > 2000):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def predict(model, texts, labels, stats):
    try:
        return model.batch_predict_entities(texts, labels, threshold=.10, flat_ner=False, multi_label=True)
    except torch.OutOfMemoryError:
        if len(texts) == 1:
            raise
    stats["oom_retries"] += 1
    gc.collect()
    torch.cuda.empty_cache()
    mid = len(texts)//2
    return predict(model, texts[:mid], labels, stats) + predict(model, texts[mid:], labels, stats)


def spans(text, model, labels, stats, windows):
    best = {}
    stats["extra_windows"] += len(windows)-1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for batch in groups(windows):
            preds = predict(model, [w for _, w in batch], labels, stats)
            for (offset, window), ents in zip(batch, preds, strict=True):
                for ent in ents:
                    a, b = ent["start"], ent["end"]
                    if not 0 <= a < b <= len(window) or ent["text"] != window[a:b]:
                        raise ValueError("Invalid window-relative span")
                    key = (offset+a, offset+b, ent["label"])
                    best[key] = max(best.get(key, 0.), float(ent["score"]))
            if torch.cuda.is_available() and any(len(w.split()) > 200 for _, w in batch):
                torch.cuda.empty_cache()
    stats["truncation_warnings"] += sum("truncated" in str(w.message) for w in caught)
    stats["spans"] += len(best)
    return [dict(start=a, end=b, label=lab, score=best[a,b,lab], text=text[a:b]) for a,b,lab in sorted(best)]
