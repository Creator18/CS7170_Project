"""No-video baselines on mc_v2 items (no model, no GPU). Accuracy per question and joint.

  random         uniform over 5 options (expected 20%)
  prior          step: most frequent step kind on TRAIN; action: P(action | answer step's kind is
                 unknown, so the option's own frequency on TRAIN); object: frequency on TRAIN
                 of the object string. Uses no session history.
  state          ORACLE HISTORY: step only; prefer started over available over blocked,
                 using the progress state built from earlier instructions (heard speech)
  same_as_last   ORACLE HISTORY: copy the previous instruction's field when it is an option
  probes (artifact checks, should sit near chance):
    position     always option 0
    longest      the option with the longest text

    python3 -I scripts/baselines_novideo.py runs/mc_v2/items.jsonl
"""
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pick(rng, scores):
    best = max(scores)
    return rng.choice([i for i, s in enumerate(scores) if s == best])


def main():
    items = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    rng = random.Random(7170)
    kinds = {}
    for p in (ROOT / "labels/recipes/graphs").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        kinds[d["dish_id"]] = {s["step_id"]: s["kind"] for s in d["steps"]}
    freq = {"step": collections.Counter(), "action": collections.Counter(), "object": collections.Counter()}
    for it in items:
        if it["split"] == "train":
            a = it["answer"]
            if a["step_id"]:
                freq["step"][kinds[it["dish_id"]][a["step_id"]]] += 1
            freq["action"][a["action"]] += 1
            if a["object"]:
                freq["object"][a["object"]] += 1

    names = ["random", "prior", "state", "same_as_last", "position", "longest"]
    acc = collections.defaultdict(collections.Counter)
    for it in items:
        joint = {n: True for n in names}
        for field, q in it["questions"].items():
            opts, ans = q["options"], q["answer_index"]
            texts = q.get("texts", [str(o) for o in opts])
            pr = {"random": rng.randrange(5), "position": 0, "longest": pick(rng, [len(t) for t in texts])}
            if field == "step":
                pr["prior"] = pick(rng, [freq["step"][kinds[it["dish_id"]][o]] for o in opts])
                st = it["state"]
                pr["state"] = pick(rng, [2 if o in st["started"] else 1 if o in st["available"] else 0 for o in opts])
            else:
                pr["prior"] = pick(rng, [freq[field][o] for o in opts])
                pr["state"] = rng.randrange(5)
            prev = it["previous"]
            key = {"step": "step_id", "action": "action", "object": "object"}[field]
            pr["same_as_last"] = pick(rng, [1 if prev and prev[key] == o else 0 for o in opts])
            for n in names:
                ok = pr[n] == ans
                joint[n] &= ok
                acc[(it["split"], field)][n] += ok
            acc[(it["split"], field)]["n"] += 1
        for n in names:
            acc[(it["split"], "joint")][n] += joint[n]
        acc[(it["split"], "joint")]["n"] += 1
    print(f"{'split':6} {'question':8} {'n':>4} " + " ".join(f"{n:>12}" for n in names))
    for sp in ("train", "dev", "test"):
        for field in ("step", "action", "object", "joint"):
            c = acc[(sp, field)]
            if c["n"]:
                print(f"{sp:6} {field:8} {c['n']:4} " + " ".join(f"{c[n] / c['n']:12.1%}" for n in names))


if __name__ == "__main__":
    main()
