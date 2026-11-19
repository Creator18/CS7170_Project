"""Map the leader's instructions to steps of the session's generic recipe graph.

An instruction is a leader thought of kind wish_for_partner (labels/thoughts).
Each one is mapped to one step of the dish's graph (labels/recipes/graphs), or
to none with a reason. The mapping is ground truth for scoring, never model input.

    python scripts/instruction_map.py build --out runs/instruction_map_v1/bundles
    python scripts/instruction_map.py check labels/recipes/instruction_map

Mapping file, one per session (labels/recipes/instruction_map/<recording_id>.json):
{
  "version": "0.1.0", "recording_id": "...", "dish_id": "...", "made_by": "...",
  "mappings": [{
    "thought_id": "...", "time": 2.574,
    "step_id": "s5" or null,
    "match": "exact" | "part_of" | "related" | "none",
    "off_graph": null | "utensil_or_item_request" | "cleaning_or_tidying" | "clarification_or_check"
                 | "not_in_generic_recipe" | "session_logistics" | "other",
    "confidence": "high" | "medium" | "low",
    "note": "short reason",
    "action": one of ACTIONS below,
    "object_id": an ingredient or equipment id from the dish graph, or null,
    "object_text": short noun phrase for the object ("the onion", "the big pot"), or null,
    "action_confidence": "high" | "medium" | "low"
  }]
}
action (added in 0.2.0): what the leader asks the helper to do, one class:
  fetch_or_hand (get, bring, pass, give), wash, peel, cut (chop, slice, dice, crush, separate),
  grate, open_or_crack (open a package, crack an egg), add (put in or on, pour in, season),
  mix (stir, toss, fold, flip), heat_control (heat on, off, up, down; appliance buttons; boil water),
  transfer (move to a plate or pot, drain, plate, serve), wait_or_stop (wait, stop, be careful),
  taste_or_check (try it, check, show me), clean_or_tidy (wash dishes, put back, throw away),
  cook (added in 0.3.0: cook, fry, boil or bake a food, keep cooking it; the object is the food),
  cover_or_uncover (added in 0.3.0: put on or take off a lid or cover), other.
heat_control is about the heat source itself (stove, oven, kettle, putting an empty pan on to heat);
cook is about a food being cooked. Dough work (knead, roll) is mix for kneading, other for rolling.
action_confidence (rule from 0.3.0): high = the request and its object are clear from the speech;
medium = the object or action needs the surrounding context to resolve; low = garbled speech, a
possibly wrong speaker, or a referent that is a guess.
object_id is null when the object is not in the graph (then object_text names it) or when the
action has no object (wait_or_stop); object_text is null only when there is no object at all.
match: exact = the instruction is the step; part_of = a sub-action of the step (open the
package while fetching); related = touches the step but is not doing it (stir it again, turn
down the heat on it); none = no step fits (step_id null, off_graph set).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATCH = {"exact", "part_of", "related", "none"}
OFF = {None, "utensil_or_item_request", "cleaning_or_tidying", "clarification_or_check",
       "not_in_generic_recipe", "session_logistics", "other"}
CONF = {"high", "medium", "low"}
ACTIONS = {"fetch_or_hand", "wash", "peel", "cut", "grate", "open_or_crack", "add", "mix", "heat_control",
           "transfer", "wait_or_stop", "taste_or_check", "clean_or_tidy", "cook", "cover_or_uncover", "other"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def instructions(th):
    return [t for t in th["thoughts"] if t["person_id"] == "leader" and t["kind"] == "wish_for_partner"]


def build(args):
    dishes = {s["recording_id"]: s for s in load(ROOT / "labels/recipes/dishes.json")["sessions"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((ROOT / "labels/thoughts").glob("*.json")):
        th = load(p)
        rid = th["recording_id"]
        dish = dishes[rid]
        g = load(ROOT / "labels/recipes/graphs" / f"{dish['dish_id']}.json")
        targets = {t["thought_id"] for t in instructions(th)}
        timeline = []
        for t in sorted(th["thoughts"], key=lambda t: t["time"]):
            utt = " / ".join(u["text"].strip() for u in t["source"].get("utterances", []))
            timeline.append({"thought_id": t["thought_id"], "time": round(t["time"], 1), "person": t["person_id"],
                             "kind": t["kind"], "text": t["text"], "said": utt[:300],
                             "object": t.get("object_resource_id"), "MAP_THIS": t["thought_id"] in targets})
        bundle = {"recording_id": rid, "dish_id": dish["dish_id"], "dish_name": dish["dish_name"],
                  "steps": [{k: s[k] for k in ("step_id", "component", "text", "kind", "requires", "checkpoint")}
                            for s in g["steps"]],
                  "n_to_map": len(targets), "timeline": timeline}
        (out / f"{rid}.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
        n += len(targets)
    print(f"wrote bundles to {out}; {n} instructions to map")


def check(args):
    dishes = {s["recording_id"]: s for s in load(ROOT / "labels/recipes/dishes.json")["sessions"]}
    bad, total = 0, 0
    for p in sorted((ROOT / "labels/thoughts").glob("*.json")):
        th = load(p)
        rid = th["recording_id"]
        mp = Path(args.root) / f"{rid}.json"
        if not mp.exists():
            print(f"MISSING {rid[:8]}")
            bad += 1
            continue
        text = mp.read_text(encoding="utf-8")
        m = json.loads(text)
        errs = ["contains an em dash"] if "—" in text else []
        g = load(ROOT / "labels/recipes/graphs" / f"{dishes[rid]['dish_id']}.json")
        steps = {s["step_id"] for s in g["steps"]}
        objects = {i["id"] for i in g["ingredients"]} | {e["id"] for e in g["equipment"]}
        want = {t["thought_id"] for t in instructions(th)}
        got = [x["thought_id"] for x in m["mappings"]]
        if m.get("dish_id") != dishes[rid]["dish_id"]:
            errs.append("wrong dish_id")
        if set(got) != want or len(got) != len(set(got)):
            errs.append(f"thought ids differ: missing {sorted(want - set(got))[:5]}, extra {sorted(set(got) - want)[:5]}")
        for x in m["mappings"]:
            tid = x["thought_id"]
            if x.get("match") not in MATCH or x.get("off_graph") not in OFF or x.get("confidence") not in CONF:
                errs.append(f"{tid}: bad enum")
            elif x["match"] == "none":
                if x.get("step_id") is not None or x.get("off_graph") is None:
                    errs.append(f"{tid}: match none needs step_id null and off_graph set")
            elif x.get("step_id") not in steps or x.get("off_graph") is not None:
                errs.append(f"{tid}: step_id {x.get('step_id')} not in graph, or off_graph set on a match")
            if args.actions:
                if x.get("action") not in ACTIONS or x.get("action_confidence") not in CONF:
                    errs.append(f"{tid}: bad or missing action / action_confidence")
                if x.get("object_id") is not None and x["object_id"] not in objects:
                    errs.append(f"{tid}: object_id {x['object_id']} not in graph")
                if x.get("object_id") is not None and not x.get("object_text"):
                    errs.append(f"{tid}: object_id set but object_text empty")
                if "object_text" not in x or "object_id" not in x:
                    errs.append(f"{tid}: missing object fields")
        total += len(got)
        if errs:
            bad += 1
            print(f"FAIL {rid[:8]}")
            for e in errs[:10]:
                print("   ", e)
    print(f"{total} mappings checked, {bad} session files with problems")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("root")
    c.add_argument("--actions", action="store_true", help="also require the 0.2.0 action and object fields")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else check(a)


if __name__ == "__main__":
    main()
