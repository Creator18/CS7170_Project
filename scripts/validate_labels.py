"""Validate coordination label files against the schema and check cross-references.

Usage:
    python scripts/validate_labels.py LABELS.json [LABELS.json ...]
    python scripts/validate_labels.py LABELS.json --comind data/comind/v1/annotations

JSON Schema covers field types and per-event rules. This script adds what a
schema cannot express: references that must resolve, time ordering, the task
hierarchy, and (with --comind) that every comind_ref points at a real,
non-null CoMind event in the same recording.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "coordination_labels.schema.json"
PARENT_LEVEL = {"step": "goal", "substep": "step"}
COMIND_FILES = {
    "handover": "dataset_handover_consolidated.json",
    "scoia": "dataset_scoia_consolidated.json",
    "joint_attention": "dataset_joint_attention_consolidated.json",
}


def load_comind(ann_dir):
    return {k: json.loads((Path(ann_dir) / f).read_text(encoding="utf-8"))["data"]
            for k, f in COMIND_FILES.items()}


def check(doc, comind=None):
    errors, warnings = [], []
    err = errors.append

    persons = {p["person_id"] for p in doc["persons"]}
    passes = {p["pass_id"] for p in doc["annotation_passes"]}
    resources = {r["resource_id"] for r in doc["resources"]}
    tasks = {t["task_id"]: t for t in doc["tasks"]}
    actions = {a["action_id"]: a for a in doc["actions"]}
    events = {e["event_id"]: e for e in doc["coordination_events"]}
    thoughts = {t["thought_id"]: t for t in doc["thoughts"]}

    for name, items, key in (("task", doc["tasks"], "task_id"), ("action", doc["actions"], "action_id"),
                             ("event", doc["coordination_events"], "event_id"),
                             ("thought", doc["thoughts"], "thought_id"),
                             ("resource", doc["resources"], "resource_id")):
        ids = [x[key] for x in items]
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            err(f"duplicate {name} id {dup}")

    lo, hi = (doc["excerpt"]["start_time"], doc["excerpt"]["end_time"]) if "excerpt" in doc else (0.0, float("inf"))

    def span(where, a, b):
        if a > b:
            err(f"{where}: start_time {a} is after end_time {b}")
        if a < lo or b > hi:
            err(f"{where}: [{a}, {b}] lies outside the excerpt [{lo}, {hi}]")

    def ref(where, kind, value, pool):
        if value is not None and value not in pool:
            err(f"{where}: unknown {kind} '{value}'")

    for t in doc["tasks"]:
        w = f"task {t['task_id']}"
        span(w, t["start_time"], t["end_time"])
        ref(w, "pass", t["pass_id"], passes)
        for p in t["person_ids"]:
            ref(w, "person", p, persons)
        if t["level"] == "goal":
            if t["parent_id"] is not None:
                err(f"{w}: a goal cannot have a parent")
            continue
        parent = tasks.get(t["parent_id"])
        if parent is None:
            err(f"{w}: parent '{t['parent_id']}' not found")
            continue
        if parent["level"] != PARENT_LEVEL[t["level"]]:
            err(f"{w}: a {t['level']} must sit under a {PARENT_LEVEL[t['level']]}, not a {parent['level']}")
        if t["start_time"] < parent["start_time"] or t["end_time"] > parent["end_time"]:
            err(f"{w}: [{t['start_time']}, {t['end_time']}] extends outside its parent {parent['task_id']}")

    for a in doc["actions"]:
        w = f"action {a['action_id']}"
        span(w, a["start_time"], a["end_time"])
        ref(w, "person", a["person_id"], persons)
        ref(w, "task", a["task_id"], tasks)
        ref(w, "resource", a.get("object_resource_id"), resources)
        ref(w, "pass", a["pass_id"], passes)
        own = a.get("intention_thought_id")
        ref(w, "intention thought", own, thoughts)
        if own in thoughts and thoughts[own]["person_id"] != a["person_id"]:
            err(f"{w}: intention_thought_id {own} belongs to {thoughts[own]['person_id']}, not {a['person_id']}")
        inf = a.get("influence")
        if inf:
            ref(w + " influence", "person", inf["partner"], persons)
            if inf["partner"] == a["person_id"]:
                err(f"{w}: influenced by its own actor")
            ref(w + " influence", "event", inf.get("event_id"), events)
            trg = inf["trigger"]
            ref(w + " influence trigger", "person", trg.get("person_id"), persons)
            ref(w + " influence trigger", "action", trg.get("action_id"), actions)
            tid = trg.get("thought_id")
            ref(w + " influence trigger", "thought", tid, thoughts)
            if tid in thoughts and thoughts[tid]["person_id"] != inf["partner"]:
                err(f"{w}: influencing thought {tid} belongs to {thoughts[tid]['person_id']}, not the partner {inf['partner']}")

    for t in doc["thoughts"]:
        w = f"thought {t['thought_id']}"
        span(w, t["time"], t["time"])
        ref(w, "person", t["person_id"], persons)
        ref(w, "person", t.get("about_person"), persons)
        ref(w, "resource", t.get("object_resource_id"), resources)
        ref(w, "pass", t["pass_id"], passes)
        if t["known_to"] != [t["person_id"]]:
            err(f"{w}: known_to must be [{t['person_id']}] (a thought is known only to its owner)")
        for aid in t["led_to_action_ids"]:
            ref(w, "action", aid, actions)
        for u in t["source"]["utterances"]:
            ref(w + " utterance", "mic", u["mic"], persons)
            if u["speaker_id"] not in ("unknown", t["person_id"]):
                err(f"{w}: utterance attributed to {u['speaker_id']} but the thought belongs to {t['person_id']}")
        if t["referent_resolution"] == "needs_video" and "[" not in t["text"]:
            warnings.append(f"{w}: needs_video but no [placeholder] in the text")

    for e in doc["coordination_events"]:
        w = f"event {e['event_id']}"
        span(w, e["start_time"], e["end_time"])
        ref(w, "person", e["actor"], persons)
        ref(w, "person", e["partner"], persons)
        if e["actor"] == e["partner"]:
            err(f"{w}: actor and partner are the same person")
        ref(w, "pass", e["pass_id"], passes)
        ref(w, "action", e.get("response_action_id"), actions)
        trg = e["trigger"]
        ref(w + " trigger", "person", trg.get("person_id"), persons)
        ref(w + " trigger", "action", trg.get("action_id"), actions)
        ref(w + " trigger", "thought", trg.get("thought_id"), thoughts)
        if trg.get("time") is not None and trg["time"] > e["end_time"]:
            err(f"{w}: trigger at {trg['time']} comes after the event ends")
        for k, v in e["task_context"].items():
            ref(f"{w} task_context.{k}", "task", v, tasks)
        for r in e.get("related_event_ids", []):
            ref(w, "related event", r, events)
        for s in e.get("speech", []):
            ref(w + " speech", "mic", s["mic"], persons)
            if s["speaker_id"] != "unknown":
                ref(w + " speech", "speaker", s["speaker_id"], persons)
            if s["start_time"] > s["end_time"]:
                err(f"{w}: speech '{s['text'][:30]}' has start after end")
        res = e.get("resource")
        if res:
            ref(w, "resource", res["resource_id"], resources)
            ref(w, "person", res.get("yielded_by"), persons)
        act = actions.get(e.get("response_action_id"))
        if act and act["person_id"] != e["actor"]:
            err(f"{w}: response action {act['action_id']} belongs to {act['person_id']}, not the actor {e['actor']}")
        if act and e["type"] in ("response", "handover") and act["partner_influence"] != "influenced":
            err(f"{w}: response action {act['action_id']} is a {e['type']} but partner_influence is '{act['partner_influence']}'")

    for p in doc["persons"]:
        if not p["side_mapping_verified"]:
            warnings.append(f"person {p['person_id']}: comind_side '{p['comind_side']}' not yet verified against video")

    if comind is not None:
        rid = doc["recording_id"]
        for kind, items in (("action", doc["actions"]), ("event", doc["coordination_events"])):
            for x in items:
                cr = x.get("comind_ref")
                if not cr:
                    continue
                entry = comind[cr["file"]].get(rid, {}).get(cr["key"], "MISSING")
                if entry == "MISSING":
                    err(f"{kind} {x.get('action_id') or x.get('event_id')}: comind_ref {cr} not found in recording {rid}")
                elif entry is None:
                    err(f"{kind} {x.get('action_id') or x.get('event_id')}: comind_ref {cr} is a hidden (null) test-set label")
    return errors, warnings


def is_implicit(e):
    return e["cue_channels"] == ["none"] and e["partner_dependency"] == "yes"


def is_implicit_action(a):
    inf = a.get("influence")
    return a["partner_influence"] == "influenced" and bool(inf) and inf["cue_channels"] == ["none"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--comind", help="path to CoMind's annotations/ folder, to resolve comind_ref")
    args = ap.parse_args()

    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")),
                                                format_checker=jsonschema.FormatChecker())
    comind = load_comind(args.comind) if args.comind else None
    failed = 0
    for f in args.files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        schema_errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
        print(f"\n{f}")
        if schema_errors:
            failed += 1
            for e in schema_errors:
                print(f"  SCHEMA  {'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}")
            continue
        errors, warnings = check(doc, comind)
        for e in errors:
            print(f"  ERROR   {e}")
        for w in warnings:
            print(f"  WARN    {w}")
        ev, acts, th = doc["coordination_events"], doc["actions"], doc["thoughts"]
        by_type = {t: sum(e["type"] == t for e in ev) for t in ("response", "handover", "resource_conflict")}
        infl = {k: sum(a["partner_influence"] == k for a in acts) for k in ("influenced", "none", "unclear", "not_assessed")}
        kinds = {}
        for t in th:
            kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
        print(f"  {len(doc['tasks'])} tasks | {len(acts)} actions {infl}, {sum(map(is_implicit_action, acts))} implicit")
        print(f"  {len(ev)} events {by_type}, {sum(map(is_implicit, ev))} implicit | {len(th)} thoughts {kinds}")
        print("  OK" if not errors else f"  FAILED ({len(errors)} errors)")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
