"""Tag each thought of one person with heard_dependency (schema 0.2.1): could this thought exist if its owner
had never heard the partner? A model that must not hear the partner should only get 'independent' thoughts.

    view   print a session's thoughts as one timeline, the tagged person's numbered (H000, H001, ...)
           and the partner's as context, for an annotator to read
    apply  read one answer file per session and write the tags into the label files

Answer file (one per session, named <first 8 chars of recording id>.txt): one line per thought that is
NOT independent, "H012 <code>[confidence]". Every numbered thought not listed is independent.
Lines starting with # are comments.

    a acknowledges_partner      n answers_partner          q asks_about_partner_speech
    s restates_partner_speech   i uses_info_from_partner_speech
    r reacts_to_partner_speech  u unclear
    confidence suffix h, m or l (default m), e.g. "H012 al"

    python scripts/tag_heard_dependency.py view  --labels labels/thoughts --session 0542e1ea --out view.txt
    python scripts/tag_heard_dependency.py apply --labels labels/thoughts --answers runs/heard_dependency_v1/answers
"""
import argparse
import json
import re
from pathlib import Path

CODES = {"a": "acknowledges_partner", "n": "answers_partner", "q": "asks_about_partner_speech",
         "s": "restates_partner_speech", "i": "uses_info_from_partner_speech",
         "r": "reacts_to_partner_speech", "u": "unclear"}
CONF = {"h": "high", "m": "medium", "l": "low"}
KIND = {"own_intention": "OI", "joint_intention": "JI", "wish_for_partner": "WP", "belief": "BE",
        "question": "QU", "acknowledgement": "AC"}
PASS_ID = "p_heard"
SCHEMA_VERSION = "0.2.1"


def find_file(labels, session):
    hits = sorted(Path(labels).glob(f"{session}*.json"))
    if len(hits) != 1:
        raise SystemExit(f"{session}: expected one label file, found {len(hits)}")
    return hits[0]


def numbered(doc, person):
    """The tagged person's thoughts in a stable order; H-numbers index into this list."""
    return sorted((t for t in doc["thoughts"] if t["person_id"] == person), key=lambda t: (t["time"], t["thought_id"]))


def view(args):
    doc = json.loads(find_file(args.labels, args.session).read_text(encoding="utf-8"))
    mine = {t["thought_id"]: i for i, t in enumerate(numbered(doc, args.person))}
    lines = [f"{doc['recording_id']}  tagging: {args.person}"]
    for t in sorted(doc["thoughts"], key=lambda t: (t["time"], t["thought_id"])):
        tag = f"H{mine[t['thought_id']]:03d}" if t["thought_id"] in mine else "   ."
        who = "" if t["thought_id"] in mine else t["person_id"][0].upper() + " "
        lines.append(f"{tag} {t['time']:7.1f} {KIND[t['kind']]} {who}{t['text']}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(mine)} {args.person} thoughts, {len(lines) - 1} in all")


def parse_answers(path, n):
    tags = {}
    for k, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"H(\d{3})\s+([anqsiru])([hml]?)", line)
        if not m:
            raise SystemExit(f"{path.name}:{k}: cannot parse {raw!r}")
        idx = int(m.group(1))
        if idx >= n:
            raise SystemExit(f"{path.name}:{k}: H{idx:03d} out of range (session has {n})")
        if idx in tags:
            raise SystemExit(f"{path.name}:{k}: H{idx:03d} listed twice")
        tags[idx] = (m.group(2), CONF[m.group(3) or "m"])
    return tags


def apply(args):
    total = {"independent": 0, "depends_on_partner_speech": 0, "unclear": 0}
    reasons = {}
    for ans in sorted(Path(args.answers).glob("*.txt")):
        path = find_file(args.labels, ans.stem)
        doc = json.loads(path.read_text(encoding="utf-8"))
        mine = numbered(doc, args.person)
        tags = parse_answers(ans, len(mine))
        for i, t in enumerate(mine):
            if i in tags:
                code, conf = tags[i]
                status = "unclear" if code == "u" else "depends_on_partner_speech"
                reason = CODES[code]
            else:
                status, reason, conf = "independent", "none", "medium"
            t["heard_dependency"] = {"status": status, "reason": reason, "pass_id": PASS_ID, "confidence": conf}
            total[status] += 1
            reasons[reason] = reasons.get(reason, 0) + 1
        doc["annotation_passes"] = [p for p in doc["annotation_passes"] if p["pass_id"] != PASS_ID] + [{
            "pass_id": PASS_ID, "layer": "heard_dependency", "annotator_id": "claude-opus-5-5", "audio": "on",
            "tool": "scripts/tag_heard_dependency.py",
            "notes": f"Tags {args.person} thoughts only, from the thought text and the surrounding thoughts of both people. Review a sample by hand."}]
        doc["schema_version"] = SCHEMA_VERSION
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": total, "reason": reasons}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view")
    v.add_argument("--labels", default="labels/thoughts")
    v.add_argument("--session", required=True)
    v.add_argument("--person", default="helper")
    v.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--labels", default="labels/thoughts")
    a.add_argument("--answers", required=True)
    a.add_argument("--person", default="helper")
    args = ap.parse_args()
    {"view": view, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
