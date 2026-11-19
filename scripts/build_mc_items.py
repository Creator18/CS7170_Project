"""Build multiple-choice items for "predict the leader's next instruction" (mc_v2).

One item per leader instruction (labels/recipes/instruction_map, v0.3.0). The model input
window ends at the instruction's utterance start (cutoff_time).

mc_v2 asks up to three questions per item, each with 5 options of ONE field:
  step    5 distinct recipe steps          (only when the instruction maps to a step)
  action  5 distinct action classes
  object  5 distinct object names          (only when the instruction has an object)
An item is jointly correct when every question it has is answered correctly.
mc_v1 offered 5 (step, action, object) triples; its distractors were built by changing one
field of the answer, so the answer shared fields with most options, and a probe that picks
the option overlapping most with the others scored 88% on test. Kept in runs/mc_v1/.

Progress state at the cutoff (an approximation until real progress labels exist):
  started   = steps mapped by an earlier instruction of this session, plus their graph ancestors
  available = not started, every prerequisite started
  blocked   = the rest
Only instructions strictly before the cutoff are used. The state comes from earlier speech, so
anything using it is an oracle-history condition, not a muted one.

Distractors (from the same session's graph and labels, never from later instructions):
  step    2 available, 1 started, 1 blocked (any pool short is filled from the others)
  action  the most frequent actions for the answer step's kind on TRAIN, then random
  object  objects of the answer step, then of available steps, then the session pool
          (graph objects plus off-graph objects named in this session's labels)

    python3 -I scripts/build_mc_items.py --out runs/mc_v2/items.jsonl
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 7170
ACTIONS = ["fetch_or_hand", "wash", "peel", "cut", "grate", "open_or_crack", "add", "mix", "heat_control",
           "transfer", "wait_or_stop", "taste_or_check", "clean_or_tidy", "cook", "cover_or_uncover", "other"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm(s):
    s = re.sub(r"^(the|a|an|some|more)\s+", "", (s or "").strip().lower())
    return re.sub(r"\s+", " ", s)


def take(rng, pool, k, exclude):
    pool = [p for p in dict.fromkeys(pool) if p not in exclude]
    rng.shuffle(pool)
    return pool[:k]


def question(rng, answer, tiers, field):
    """answer plus 4 distractors drawn tier by tier: tiers = [(name, candidates, k), ...]."""
    chosen, src = [answer], ["answer"]
    for name, cands, k in tiers:
        for c in take(rng, cands, k, set(chosen)):
            chosen.append(c)
            src.append(name)
    for name, cands, _ in tiers:  # top up from any tier if short
        if len(chosen) >= 5:
            break
        for c in take(rng, cands, 5 - len(chosen), set(chosen)):
            chosen.append(c)
            src.append(name + "_fill")
    if len(chosen) < 5:
        return None
    chosen, src = chosen[:5], src[:5]
    order = list(range(5))
    rng.shuffle(order)
    return {"field": field, "options": [chosen[i] for i in order], "sources": [src[i] for i in order],
            "answer_index": order.index(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    sp = load(ROOT / "splits/thoughts_v1.json")
    split = {r: k for k in ("train", "dev", "test") for r in sp[k]}
    dishes = {s["recording_id"]: s for s in load(ROOT / "labels/recipes/dishes.json")["sessions"]}

    sessions = {}
    for p in sorted((ROOT / "labels/recipes/instruction_map").glob("*.json")):
        m = load(p)
        th = {t["thought_id"]: t for t in load(ROOT / "labels/thoughts" / p.name)["thoughts"]}
        sessions[m["recording_id"]] = (m, th, load(ROOT / "labels/recipes/graphs" / f"{m['dish_id']}.json"))

    kind_actions = collections.defaultdict(collections.Counter)
    all_actions = collections.Counter()
    for rid, (m, th, g) in sessions.items():
        if split.get(rid) != "train":
            continue
        kinds = {s["step_id"]: s["kind"] for s in g["steps"]}
        for x in m["mappings"]:
            all_actions[x["action"]] += 1
            if x["step_id"]:
                kind_actions[kinds[x["step_id"]]][x["action"]] += 1

    counts = collections.Counter()
    with out.open("w", encoding="utf-8") as f:
        for rid, (m, th, g) in sorted(sessions.items()):
            steps = {s["step_id"]: s for s in g["steps"]}
            names = {i["id"]: norm(i["name"]) for i in g["ingredients"]} | {e["id"]: norm(e["name"]) for e in g["equipment"]}

            def ancestors(sid, acc):
                for r in steps[sid]["requires"]:
                    if r not in acc:
                        acc.add(r)
                        ancestors(r, acc)
                return acc

            def obj(x):
                return names[x["object_id"]] if x["object_id"] else (norm(x["object_text"]) or None)

            def step_objs(sid):
                return [names[i] for i in steps[sid]["ingredients"] + steps[sid]["equipment"]]

            pool = sorted(set(names.values()) | {norm(x["object_text"]) for x in m["mappings"]
                                                 if not x["object_id"] and x["object_text"]} - {""})
            xs = []
            for x in m["mappings"]:
                utts = th[x["thought_id"]]["source"].get("utterances") or []
                xs.append((min([u["start_time"] for u in utts] or [x["time"]]), x))
            xs.sort(key=lambda t: t[0])

            for k, (cutoff, x) in enumerate(xs):
                prior = [y for c, y in xs[:k] if c < cutoff]
                started = set()
                for y in prior:
                    if y["step_id"]:
                        started.add(y["step_id"])
                        ancestors(y["step_id"], started)
                available = [s for s in steps if s not in started and all(r in started for r in steps[s]["requires"])]
                blocked = [s for s in steps if s not in started and s not in available]
                qs = {}
                if x["step_id"]:
                    qs["step"] = question(rng, x["step_id"], [("available", available, 2), ("started", sorted(started), 1),
                                                             ("blocked", blocked, 1)], "step")
                kind = steps[x["step_id"]]["kind"] if x["step_id"] else None
                freq = [a for a, _ in (kind_actions[kind] if kind else all_actions).most_common()]
                qs["action"] = question(rng, x["action"], [("frequent_for_kind", freq[:6], 3), ("random", ACTIONS, 1)], "action")
                ans_obj = obj(x)
                if ans_obj:
                    near = step_objs(x["step_id"]) if x["step_id"] else []
                    avail_objs = [o for s in available for o in step_objs(s)]
                    qs["object"] = question(rng, ans_obj, [("same_step", near, 2), ("available_steps", avail_objs, 1),
                                                          ("session_pool", pool, 1)], "object")
                if any(q is None for q in qs.values()):
                    counts["dropped_short_question"] += 1
                    qs = {kk: q for kk, q in qs.items() if q is not None}
                for q in qs.values():
                    if q["field"] == "step":
                        q["texts"] = [steps[s]["text"] for s in q["options"]]
                prev = prior[-1] if prior else None
                item = {
                    "item_id": f"{rid[:8]}_{x['thought_id']}", "recording_id": rid, "split": split.get(rid),
                    "dish_id": m["dish_id"], "dish_name": dishes[rid]["dish_name"], "thought_id": x["thought_id"],
                    "cutoff_time": round(cutoff, 3), "match": x["match"], "off_graph": x["off_graph"],
                    "label_confidence": x["confidence"], "action_confidence": x["action_confidence"],
                    "answer": {"step_id": x["step_id"], "action": x["action"], "object": ans_obj},
                    "questions": qs,
                    "state": {"started": sorted(started), "available": sorted(available)},
                    "previous": ({"step_id": prev["step_id"], "action": prev["action"], "object": obj(prev)} if prev else None),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[(item["split"], "+".join(sorted(qs)))] += 1
    for k, v in sorted(counts.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()
