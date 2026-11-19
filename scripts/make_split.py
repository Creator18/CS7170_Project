"""Fix a session-level train / dev / test split of labels/thoughts, so that a test set exists that
no training run, prompt design or model choice ever touches.

Splitting is by recording (session): every action, event and thought of a session lands in one part.
CoMind does not ship participant or kitchen ids, so a person or kitchen can still appear on both
sides; the split file records that limitation.

Test and dev sessions are drawn only from sessions whose speaker attribution was mostly confident, because
the "why" ground truth for the trace-back task is the leader thought linked to each helper action,
and that thought is only as good as its speaker attribution. Among eligible sessions, many seeded
random partitions are tried and the one whose dev and test shares best match the targets on four
counts (sessions, helper actions linked to a leader thought, speech-only cued helper actions,
gesture-cued helper actions) is kept. The seed makes the result reproducible.

    python scripts/make_split.py --labels labels/thoughts --out splits/thoughts_v1.json
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 7170
TRIES = 20000
TARGET = {"test": 0.20, "dev": 0.12}

# Sessions whose speakers were attributed mostly at low confidence or with labels crossing
# throughout (see labels/thoughts/README.md). They may be trained on but are kept out of dev and test.
NOT_TESTABLE = {
    "833eac29": "both mics label both people; speakers by content, mostly low confidence",
    "bb657948": "labels cross throughout; most groups low confidence",
    "d90d7d88": "both mics label both people S00",
    "b1446007": "leader mic labels both people",
    "9db1ea2e": "leader mic labels both people; helper role changes hands mid-session",
    "199d1b52": "one label covers both people",
    "1df816f6": "one label covers both people",
    "222861ac": "one label covers both people",
    "5f8dc440": "one label covers both people",
    "6473136b": "one label covers both people",
    "845323e6": "leader mic has one label for both people",
    "16b9e645": "dominant label on both mics is the helper",
}


def session_stats(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    leader_links = {a for t in d["thoughts"] if t["person_id"] == "leader"
                    for a in (t.get("led_to_action_ids") or [])}
    s = {"linked": 0, "speech_only": 0, "gesture": 0, "helper_actions": 0, "thoughts": len(d["thoughts"])}
    for a in d["actions"]:
        if a["person_id"] != "helper":
            continue
        s["helper_actions"] += 1
        ch = set(a.get("influence", {}).get("cue_channels") or [])
        s["speech_only"] += ch == {"speech"}
        s["gesture"] += "gesture" in ch
        s["linked"] += a["action_id"] in leader_links
    return d["recording_id"], s


def cost(parts, stats, totals):
    c = 0.0
    for name, share in TARGET.items():
        ids = parts[name]
        c += (len(ids) / len(stats) - share) ** 2
        for k in ("linked", "speech_only", "gesture"):
            c += (sum(stats[i][k] for i in ids) / totals[k] - share) ** 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels/thoughts")
    ap.add_argument("--out", default="splits/thoughts_v1.json")
    args = ap.parse_args()

    stats = dict(session_stats(p) for p in sorted(Path(args.labels).glob("*.json")))
    ids = sorted(stats)
    totals = {k: sum(s[k] for s in stats.values()) for k in ("linked", "speech_only", "gesture")}
    eligible = [i for i in ids if i[:8] not in NOT_TESTABLE]
    n_test, n_dev = round(TARGET["test"] * len(ids)), round(TARGET["dev"] * len(ids))

    rng = random.Random(SEED)
    best = None
    for _ in range(TRIES):
        test = rng.sample(eligible, n_test)
        dev = rng.sample([i for i in eligible if i not in test], n_dev)
        parts = {"test": sorted(test), "dev": sorted(dev),
                 "train": sorted(i for i in ids if i not in test and i not in dev)}
        c = cost(parts, stats, totals)
        if best is None or c < best[0]:
            best = (c, parts)
    parts = best[1]

    summary = {}
    for name, members in parts.items():
        summary[name] = {"sessions": len(members),
                         **{k: sum(stats[i][k] for i in members) for k in ("helper_actions", "linked", "speech_only", "gesture", "thoughts")}}
    digest = hashlib.sha256(json.dumps(parts["test"]).encode()).hexdigest()
    out = {
        "split_id": "thoughts_v1",
        "seed": SEED,
        "unit": "recording_id",
        "rules": [
            "Never train, tune prompts, choose models or inspect per-item outputs on test.",
            "Use dev for every development decision; score test once per finished method.",
            "Test 'why' ground truth must be hand-reviewed before it is scored.",
        ],
        "limitations": [
            "CoMind ships no participant or kitchen ids, so people and kitchens may repeat across parts.",
            "Test and dev exclude the low-confidence speaker sessions, so they are cleaner than train.",
            "eb784196 has no thoughts file and is in no part.",
        ],
        "not_testable": {k: v for k, v in NOT_TESTABLE.items()},
        "test_sha256": digest,
        "summary": summary,
        **parts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("test sha256", digest)


if __name__ == "__main__":
    main()
