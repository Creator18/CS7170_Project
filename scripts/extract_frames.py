"""Extract the muted visual context for each mc_v2 item: frames from both egocentric views
in a window that ends just before the leader starts speaking.

For an item with cutoff time c (the earliest start of the instruction's utterances), frames are
taken at c - gap - (n - 1) / fps, ..., c - gap, so the last frame is `gap` seconds before speech
starts. Times below 0 are clamped to 0 (early items get a shorter real window, recorded in the
index). Video and transcripts share one clock (the *_trimmed_sync files).

Each (session, view) is decoded by one worker, seeking only when the next target is far ahead, so
overlapping windows are decoded once. Output: OUT/<item_id>/{leader,helper}_<k>.jpg plus
OUT/index.jsonl with the frame times per item.

    python -I scripts/extract_frames.py --items items.jsonl --videos comind/v1/recordings --out frames_w32 --workers 8
"""
import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path

VIEWS = ("leader", "helper")


def targets(cutoff, n, fps, gap):
    return [max(0.0, cutoff - gap - (n - 1 - k) / fps) for k in range(n)]


def do_session(job):
    rid, v, items, a = job
    import av
    from PIL import Image
    reqs = sorted((t, it["item_id"], k) for it in items
                  for k, t in enumerate(targets(it["cutoff_time"], a.frames, a.fps, a.gap)))
    # resume: skip frames already written by an earlier (timed-out) run
    reqs = [r for r in reqs if not (Path(a.out) / r[1] / f"{v}_{r[2]:02d}.jpg").exists()]
    skipped = len(items) * a.frames - len(reqs)
    done = 0
    c = av.open(f"{a.videos}/{rid}/mp4s/{v}_trimmed_sync.mp4")
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    tb = float(s.time_base)
    i = 0
    while i < len(reqs):
        c.seek(int(max(0.0, reqs[i][0] - 1.0) / tb), stream=s, backward=True, any_frame=False)
        prev, i0 = None, i
        for f in c.decode(s):
            ft = f.pts * tb
            # each target gets the latest frame at or before it (or the first decoded frame)
            while i < len(reqs) and ft > reqs[i][0] + 1e-6:
                _save(prev if prev is not None else f, reqs[i], v, a, Image)
                i += 1
                done += 1
            prev = f
            # all done, or a far gap after progress: seek instead of decoding through it.
            # (Never re-seek before reaching the seek target: with long keyframe intervals that loops forever.)
            if i == len(reqs) or (i > i0 and reqs[i][0] - ft > 8.0):
                break
        else:
            while i < len(reqs):  # targets past the last frame
                _save(prev, reqs[i], v, a, Image)
                i += 1
                done += 1
    c.close()
    return rid, v, done, skipped


def _save(frame, req, view, a, Image):
    t, item_id, k = req
    d = Path(a.out) / item_id
    d.mkdir(parents=True, exist_ok=True)
    img = frame.to_image().resize((a.size, a.size), Image.BICUBIC)
    tmp = d / f".{view}_{k:02d}.tmp"
    img.save(tmp, format="JPEG", quality=90)
    tmp.rename(d / f"{view}_{k:02d}.jpg")  # atomic, so a killed job never leaves a half-written frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between the last frame and speech start")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    items = [it for it in items if it["split"] in a.splits.split(",")]
    by = collections.defaultdict(list)
    for it in items:
        by[it["recording_id"]].append(it)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "index.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            ts = targets(it["cutoff_time"], a.frames, a.fps, a.gap)
            f.write(json.dumps({"item_id": it["item_id"], "cutoff_time": it["cutoff_time"], "times": [round(t, 3) for t in ts],
                                "window": round(ts[-1] - ts[0], 3)}) + "\n")
    (Path(a.out) / "config.json").write_text(json.dumps(vars(a), indent=1))
    # one unit per (session, view); longest sessions first so the pool stays busy
    jobs = sorted(((rid, v, its, a) for rid, its in by.items() for v in VIEWS), key=lambda j: -len(j[2]))
    total = 0
    with mp.Pool(a.workers) as pool:
        for rid, v, n, sk in pool.imap_unordered(do_session, jobs):
            total += n + sk
            print(f"{rid[:8]} {v} {n} new, {sk} already there", flush=True)
    expect = len(items) * a.frames * len(VIEWS)
    print(f"done: {len(items)} items, {total} of {expect} frames")
    assert total == expect


if __name__ == "__main__":
    main()
