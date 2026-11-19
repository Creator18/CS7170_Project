"""Convert CoMind's training annotations into seed label files (schema 0.1.1).

Usage:
    python scripts/comind_to_labels.py --comind data/comind/v1 --out labels/seed

One file per training recording. What is converted, and how:

  CoMind handover      -> a 'handover' event plus the sender's 'pass' action
  CoMind cued action   -> a 'response' event plus the helper's action
  transcripts          -> speech in the 10 s before each event, from both mics,
                          speaker left as 'unknown' (WhisperX labels are anonymous)

What is NOT produced, because it needs a person watching the video:
  - the task layer. Each file gets one placeholder goal, 'g0', so references
    resolve. Replace it with real goals, steps and substeps.
  - resource conflicts, and implicit events CoMind did not label.
  - verification that left = leader. Every file has side_mapping_verified false.

Joint attention is not converted: it is a shared-attention state, not a
coordination event in this schema.

Timing, from CoMind's annotator guide. Cued action: start_time is just after the
cue is complete, end_time is just before the action becomes obvious, so the
action begins at end_time. Handover: the segment from the prediction frame
(before any reach) to the handover; the exact end semantics are not in the
supplementary we read, so treat it as approximate.
"""

import argparse
import json
import re
from pathlib import Path

SCHEMA_VERSION = "0.2.0"
VERBS = {"grab", "receive", "pass", "place", "open", "add", "turn on", "cut", "pour", "press", "move", "rotate",
         "turn off", "scoop", "mix", "dispose", "apply", "clean", "close", "roll", "stretch",
         "wait", "step_aside", "reach", "look", "other"}
FIXTURE_L3 = {"fixture", "storage", "appliance"}
SIDE_TO_PERSON = {"left": "leader", "right": "helper"}       # inferred, unverified
FLOW = {"ltr": ("left", "right"), "rtl": ("right", "left")}   # (sender side, receiver side)
HANDOVER_CUE = {"verbal": "speech", "gestural": "gesture", "implicit": "none"}
SCOIA_CUE = {"A": ["speech"], "G": ["gesture"], "AG": ["speech", "gesture"], "O": ["none"], None: ["unknown"]}
WINDOW_S = 10.0
CUE_ORDER = ["speech", "gesture", "none", "unknown"]   # one canonical order for cue_channels


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "unnamed"


def r3(x):
    return round(float(x), 3)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def imported_action_fields(partner, trigger, cues, event_id):
    """Fields every CoMind-derived action gets: it was influenced by the partner (that is
    what CoMind labelled), and nothing else about it has been assessed yet."""
    return {"relevance": "not_assessed", "partner_influence": "influenced",
            "influence": {"partner": partner, "trigger": dict(trigger), "cue_channels": list(cues), "event_id": event_id},
            "intention_thought_id": None, "predictable_silently": "not_assessed", "detection": "comind_import"}


def speech_window(transcripts, t1):
    """Transcript segments overlapping [t1 - WINDOW_S, t1], from both mics."""
    t0, out = t1 - WINDOW_S, []
    for mic, segs in transcripts.items():
        for s in segs:
            text = (s.get("text") or "").strip()
            if text and s["end"] > t0 and s["start"] < t1:
                out.append({"text": text, "start_time": r3(s["start"]), "end_time": r3(s["end"]), "mic": mic,
                            "whisper_speaker": s.get("speaker"), "speaker_id": "unknown",
                            "speaker_attribution": "unknown"})
    return sorted(out, key=lambda s: (s["start_time"], s["mic"]))


class Resources:
    def __init__(self):
        self.items = {}

    def add(self, l1, l3):
        rid = slug(l1)
        if rid not in self.items:
            kind = "fixture" if (l3 or "").lower() in FIXTURE_L3 else "object"
            self.items[rid] = {"resource_id": rid, "type": kind, "subtype": l3 or "", "name": l1,
                               "comind_instance": None}
        return rid


def convert(rid, handovers, scoias, transcripts):
    res, actions, events = Resources(), [], []

    for key, h in sorted(handovers.items()):
        if h is None:
            continue
        if h["delivering_flow"] not in FLOW or h["initiator"] not in SIDE_TO_PERSON:
            raise ValueError(f"{rid} handover {key}: unexpected flow/initiator {h['delivering_flow']}/{h['initiator']}")
        s_side, r_side = FLOW[h["delivering_flow"]]
        sender, receiver = SIDE_TO_PERSON[s_side], SIDE_TO_PERSON[r_side]
        initiator = SIDE_TO_PERSON[h["initiator"]]
        types = h.get("initiation_type") or []
        cues = sorted({HANDOVER_CUE[t] for t in types if t in HANDOVER_CUE}, key=CUE_ORDER.index) or ["unknown"]
        if "none" in cues and len(cues) > 1:
            cues = [c for c in cues if c != "none"]
        if initiator == receiver:
            kind = ("partner_speech" if "speech" in cues else "partner_gesture" if "gesture" in cues else "unclear")
            trg = {"kind": kind, "person_id": receiver, "time": None, "action_id": None,
                   "description": f"CoMind: receiver initiated ({'+'.join(types) or 'no type'}). Cue time not annotated."}
        else:
            trg = {"kind": "partner_need_inferred", "person_id": receiver, "time": None, "action_id": None,
                   "description": f"CoMind: sender initiated ({'+'.join(types) or 'no type'}), i.e. offered unasked."}
        obj = res.add(h["object_category_level_1"], h.get("object_category_level_3"))
        aid, eid = f"h{key}.a", f"h{key}"
        actions.append({"action_id": aid, "person_id": sender, "start_time": r3(h["start_time"]),
                        "end_time": r3(h["end_time"]), "verb": "pass", "noun": h["object_category_level_1"],
                        "object_resource_id": obj, "task_id": "g0",
                        **imported_action_fields(receiver, trg, cues, eid),
                        "comind_ref": {"file": "handover", "key": key}, "pass_id": "p0",
                        "confidence": "low" if h.get("skip") else "medium",
                        **({"notes": "CoMind skip=true"} if h.get("skip") else {})})
        events.append({"event_id": eid, "type": "handover", "start_time": r3(h["start_time"]),
                       "end_time": r3(h["end_time"]), "actor": sender, "partner": receiver,
                       "response_action_id": aid, "trigger": trg, "cue_channels": cues,
                       "speech": speech_window(transcripts, h["start_time"]), "partner_dependency": "yes",
                       "task_context": {"actor_task_id": "g0", "partner_task_id": "g0", "shared_goal_id": "g0"},
                       "resource": None, "comind_ref": {"file": "handover", "key": key}, "related_event_ids": [],
                       "visually_resolvable": "not_assessed", "first_person_account": None, "pass_id": "p0",
                       "confidence": "low" if h.get("skip") else "medium",
                       **({"notes": "CoMind skip=true"} if h.get("skip") else {})})

    for key, s in sorted(scoias.items()):
        if s is None:
            continue
        verb = (s.get("verbs") or ["other"])[0]
        noun = (s.get("nouns") or ["unknown"])[0]
        note = []
        if verb not in VERBS:
            note.append(f"CoMind verb '{verb}' mapped to other")
            verb = "other"
        if s.get("annot_type") not in SCOIA_CUE:
            raise ValueError(f"{rid} scoia {key}: unexpected annot_type {s.get('annot_type')!r}")
        cues = SCOIA_CUE[s.get("annot_type")]
        kind = {"A": "partner_speech", "AG": "partner_speech", "G": "partner_gesture"}.get(s.get("annot_type"), "unclear")
        obj = res.add(noun, (s.get("nouns_level3") or [None])[0])
        aid, eid = f"s{key}.a", f"s{key}"
        trg = {"kind": kind, "person_id": "leader", "time": r3(s["start_time"]), "action_id": None,
               "thought_id": None, "description": f"CoMind cue tag {s.get('annot_type')}; time is cue completion."}
        actions.append({"action_id": aid, "person_id": "helper", "start_time": r3(s["end_time"]),
                        "end_time": r3(s["end_time"]), "verb": verb, "noun": noun, "object_resource_id": obj,
                        "task_id": "g0", **imported_action_fields("leader", trg, cues, eid),
                        "comind_ref": {"file": "scoia", "key": key}, "pass_id": "p0",
                        "confidence": "medium", "notes": "; ".join(["onset only, duration not annotated"] + note)})
        events.append({"event_id": eid, "type": "response", "start_time": r3(s["start_time"]),
                       "end_time": r3(s["end_time"]), "actor": "helper", "partner": "leader",
                       "response_action_id": aid,
                       "trigger": trg,
                       "cue_channels": cues, "speech": speech_window(transcripts, s["start_time"]),
                       "partner_dependency": "yes",
                       "task_context": {"actor_task_id": "g0", "partner_task_id": "g0", "shared_goal_id": "g0"},
                       "resource": None, "comind_ref": {"file": "scoia", "key": key}, "related_event_ids": [],
                       "visually_resolvable": "not_assessed", "first_person_account": None, "pass_id": "p0",
                       "confidence": "medium"})

    # Link a handover and the helper's receive/grab of the same object inside it (same moment, two CoMind files).
    by_id = {e["event_id"]: e for e in events}
    noun_of = {a["action_id"]: slug(a["noun"]) for a in actions}
    for h in [e for e in events if e["type"] == "handover"]:
        for s in [e for e in events if e["type"] == "response"]:
            a = next(x for x in actions if x["action_id"] == s["response_action_id"])
            if (a["verb"] in {"receive", "grab"} and noun_of[a["action_id"]] == noun_of[h["response_action_id"]]
                    and h["start_time"] - 1.0 <= a["start_time"] <= h["end_time"] + 1.0):
                h["related_event_ids"].append(s["event_id"])
                by_id[s["event_id"]]["related_event_ids"].append(h["event_id"])

    seg_end = max((x["end"] for segs in transcripts.values() for x in segs), default=0.0)
    end = r3(max([seg_end] + [e["end_time"] for e in events] + [a["end_time"] for a in actions]) + 0.001)
    for e in events:
        for sp in e["speech"]:
            end = max(end, sp["end_time"])
    return {
        "schema_version": SCHEMA_VERSION, "recording_id": rid, "source_dataset": "comind-v1",
        "timeline": "comind_trimmed_sync", "fps": 30,
        "notes": ("SEED FILE generated by scripts/comind_to_labels.py from CoMind training annotations. "
                  "Task layer is a placeholder (g0). left = leader is unverified. Speakers are not attributed."),
        "persons": [
            {"person_id": "leader", "role": "leader", "knows_plan": True, "comind_side": "left",
             "side_mapping_verified": False},
            {"person_id": "helper", "role": "helper", "knows_plan": False, "comind_side": "right",
             "side_mapping_verified": False}],
        "annotation_passes": [{"pass_id": "p0", "layer": "coordination", "annotator_id": "comind-import",
                               "audio": "on", "tool": "scripts/comind_to_labels.py",
                               "notes": "Converted from CoMind's human annotations; CoMind annotators worked with subtitles."}],
        "resources": sorted(res.items.values(), key=lambda r: r["resource_id"]),
        "tasks": [{"task_id": "g0", "person_ids": ["leader", "helper"], "level": "goal", "parent_id": None,
                   "start_time": 0.0, "end_time": end, "label": "unlabelled_session",
                   "description": "placeholder in order to let actions reference a task until the task layer is annotated",
                   "intent": "placeholder", "relevance": "essential", "is_continued": False, "ordinal": 1,
                   "pass_id": "p0", "confidence": "low", "notes": "PLACEHOLDER. Replace with real goals, steps and substeps."}],
        "actions": sorted(actions, key=lambda a: (a["start_time"], a["action_id"])),
        "thoughts": [],
        "coordination_events": sorted(events, key=lambda e: (e["start_time"], e["event_id"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comind", required=True, help="CoMind download root, e.g. data/comind/v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.comind), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ann = root / "annotations"
    H = load(ann / "dataset_handover_consolidated.json")["data"]
    S = load(ann / "dataset_scoia_consolidated.json")["data"]
    train = (root / "split" / "train.txt").read_text().split()
    for rid in train:
        tx = {who: load(root / "recordings" / rid / "transcripts" / f"{who}_trimmed_sync_transcript.json")["segments"]
              for who in ("leader", "helper")}
        doc = convert(rid, H.get(rid, {}), S.get(rid, {}), tx)
        (out / f"{rid}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(train)} seed files to {out}")


if __name__ == "__main__":
    main()
