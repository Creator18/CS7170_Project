"""Download only the two first-person videos per CoMind session.

The official script (comind_download.py) selects whole parts, so --parts mp4 would fetch all
eight MP4s per session (1.14 TiB for the 55 training sessions). This reads the per-session
healthcheck.json manifests already saved by that script and fetches only
mp4s/leader_trimmed_sync.mp4 and mp4s/helper_trimmed_sync.mp4 (110 files, 183.6 GiB),
from the same server. Resumes partial files, verifies each file against the manifest hash,
never deletes anything except its own .part file after it has been renamed into place.

    python3 -I scripts/fetch_ego_videos.py --root /scratch/basu.ang/cs7170/comind/v1
"""
import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://comind.ethz.ch/dataset"
WANT = ("mp4s/leader_trimmed_sync.mp4", "mp4s/helper_trimmed_sync.mp4")
UA = "comind-v1-download/1.0 (+https://comind.ethz.ch)"


def sha(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, size, retries=6):
    part = Path(str(dest) + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                for chunk in iter(lambda: r.read(8 << 20), b""):
                    f.write(chunk)
            if part.stat().st_size == size:
                os.replace(part, dest)
                return
            print(f"   size {part.stat().st_size} != {size}, retrying", flush=True)
        except Exception as e:  # network errors: back off and resume
            print(f"   attempt {attempt}: {type(e).__name__}: {e}", flush=True)
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"failed after {retries} attempts: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    ids = (root / "split" / "train.txt").read_text().split()
    total = done = 0
    bad = []
    for i, rid in enumerate(ids, 1):
        man = json.loads((root / "recordings" / rid / "healthcheck.json").read_text())
        algo = man.get("hash_algorithm", "sha256")
        for e in man["files"]:
            if e["path"] not in WANT:
                continue
            dest = root / "recordings" / rid / e["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            total += e["size"]
            if not (dest.exists() and dest.stat().st_size == e["size"]):
                url = f"{BASE}/{rid}/" + "/".join(urllib.parse.quote(s) for s in e["path"].split("/"))
                t0 = time.time()
                fetch(url, dest, e["size"])
                print(f"[{i}/{len(ids)}] {rid[:8]} {e['path']} {e['size'] / 2**30:.2f} GiB in {time.time() - t0:.0f} s", flush=True)
            ok = e.get("hash") is None or sha(dest, algo) == e["hash"]
            if not ok:
                bad.append(f"{rid}/{e['path']}")
                print(f"   HASH MISMATCH {rid[:8]} {e['path']} (kept for inspection)", flush=True)
            done += e["size"]
    print(f"done: {done / 2**30:.1f} of {total / 2**30:.1f} GiB, {len(bad)} hash mismatches", flush=True)
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()
