"""Recovery for a single whitespace-free run rejected by the original splitter."""
def split_run(text, offset, limit, overlap, count):
    if count(text) <= limit:
        return [(offset, text)]
    if len(text) < 2:
        raise ValueError(f"One character exceeds the tokenizer budget at offset {offset}")
    mid = len(text) // 2
    # Overlap is in original characters inside a run; cap it so both children shrink.
    pad = min(overlap, len(text) // 4)
    left_end, right_start = mid + (pad + 1) // 2, mid - pad // 2
    return (split_run(text[:left_end], offset, limit, overlap, count)
            + split_run(text[right_start:], offset + right_start, limit, overlap, count))


def compatible(saved, current, old_hashes):
    if saved == current:
        return True
    # Old scorers could not commit a group containing an oversized window.
    legacy = {k: v for k, v in current.items() if k != "window_repair_sha256"}
    legacy.update(old_hashes)
    return saved == legacy
