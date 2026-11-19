"""Recast CoMind speech as private first-person thoughts, using Claude Opus 5.5.

The silent-kitchen layer: everything participants said aloud becomes what they
privately intended, wished, believed or wondered. Output follows the 'thoughts'
layer of schemas/coordination_labels.schema.json (v0.2.0).

Steps (run in order):
  build     split each session's two transcripts into time chunks; one request per chunk. No API calls.
  estimate  rough cost of the built run. Local character count; add --api to use the token-counting endpoint.
  pilot     run a few chunks one at a time, with refusal fallback, to check quality before spending more.
  submit    send all (or some) requests as one Message Batch, at half price.
  collect   wait for the batch and save the raw results.
  merge     check the model's output and write thoughts into copies of the seed label files.

  python scripts/speech_to_thoughts.py build    --comind data/comind/v1 --seeds labels/seed --run runs/thoughts_v1
  python scripts/speech_to_thoughts.py estimate --run runs/thoughts_v1
  python scripts/speech_to_thoughts.py pilot    --run runs/thoughts_v1 --session <recording_id> --chunks 3
  python scripts/speech_to_thoughts.py submit   --run runs/thoughts_v1
  python scripts/speech_to_thoughts.py collect  --run runs/thoughts_v1
  python scripts/speech_to_thoughts.py merge    --run runs/thoughts_v1 --seeds labels/seed --out labels/thoughts

pilot, submit, collect and estimate --api need the 'anthropic' package and credentials
(ANTHROPIC_API_KEY, or a profile from `ant auth login`). build and merge do not.

Batch requests cannot carry the server-side refusal fallback, so a refused chunk is
recorded in results.jsonl; rerun it with `pilot --custom-ids ...`, which does.
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00, 20.00            # $ per million tokens, standard; batch is half
BATCH_DISCOUNT = 0.5
KINDS = ["own_intention", "joint_intention", "wish_for_partner", "belief", "question", "acknowledgement"]
RESOLUTION = ["named_in_speech", "from_linked_event", "from_context", "needs_video", "not_needed"]

SYSTEM = """You are helping build a research dataset about how two people coordinate while cooking together.

The recordings come from CoMind. In every session two people cook in a real home kitchen, each wearing camera glasses with a microphone. One is the LEADER, who knows the recipe and gives instructions. The other is the HELPER, who does not know the recipe. Each person's speech was transcribed separately from their own glasses by an automatic speech recogniser, and both microphones pick up both people, so the same words usually appear twice, transcribed slightly differently (for example "sun oil" on one mic and "saw oil" on the other).

We are building a "silent kitchen" version of each session. Imagine nobody spoke at all: everything they said aloud becomes what the speaker privately thought at that moment. A model will later watch one person's video and be given only that person's own thoughts, never the partner's, so your rewrites must read as private thoughts, not as speech. Never write "I tell", "I ask", "I said", "I want to tell", or anything that implies the words were spoken.

For the chunk of dialogue you are given, do the following.

1. GROUP the target lines. Put every line that is the same spoken words, from either microphone, into one group, and merge fragments of one sentence that were split. Every target line id must appear in exactly one group. Lines marked CTX are earlier context only: use them to understand the conversation, but never put them in a group.

2. Decide the SPEAKER of each group: leader, helper, or unknown. Weigh the content and the roles (instructions and recipe knowledge suggest the leader; asking what to do suggests the helper), the flow of the conversation, which microphone captured the fuller and cleaner version (a person's own microphone is closest to their mouth), and the per-microphone speaker-label counts provided (the label that dominates a microphone is most likely that microphone's wearer; labels are not consistent between the two microphones). Report your confidence honestly and use unknown when the evidence is weak. Say briefly what decided it.

3. Decide whether to CONVERT or DROP the group. Drop talk about the recording or the researchers (for example "don't put anything in front of the camera, he said"), pure filler, unintelligible recogniser noise, and small talk unrelated to the cooking. Convert everything that bears on the cooking, including short replies that accept or reject a plan.

4. Write one or more THOUGHTS for each converted group. A single line can carry more than one, for example "you start on the chicken, I'll do the rice" is a wish for the partner and an own intention. Kinds:
   - own_intention: what the speaker is about to do. "I'm going for the knife."
   - joint_intention: something the speaker means for both of them to do. "We should plate it now."
   - wish_for_partner: what the speaker wants the partner to do. "I want the helper to pass me the salt."
   - belief: what the speaker takes to be true, including judgements. "We need more salt."
   - question: something the speaker is unsure about. "I wonder whether I should peel the potatoes."
   - acknowledgement: the speaker accepts or rejects a plan or request. "I accept doing the onions next."
   Write in the first person, present tense. Refer to the partner as "the leader" or "the helper". Keep the speaker's certainty: a question stays uncertain, a hedge stays hedged. Do not add intentions or reasons the words do not support.

5. Resolve OBJECTS. Replace "that", "it", "this" and similar with the actual object when you can tell, using in order: words in the dialogue, the CoMind events listed for this chunk (they give real objects and times), or the cooking context. Set referent_resolution to match. If you still cannot tell, keep a placeholder in square brackets, such as "[that]", and set needs_video. Never invent an object. Put the main object in the object field, or an empty string if there is none.

6. LINK a thought to a listed CoMind event when that event is the action the thought led to, for example "pass me the salt" followed by a handover of salt. Use the event id exactly as listed; otherwise use an empty string.

Examples, from a real session:

Line: leader mic, "okay you can start by the chicken get it all I'm gonna do the rice"
Speaker: leader (gives the plan, label dominant on the leader mic)
Thoughts: wish_for_partner "I want the helper to start preparing the chicken." / own_intention "I'm going to cook the rice."

Line: helper mic, "Salt, pepper."
Speaker: helper (asks for ingredients; the leader passes salt to the helper 4 s later)
Thought: wish_for_partner "I want the leader to pass me the salt and pepper." linked to the salt handover.

Line: leader mic, "Sorry, I need to get that side."
Speaker: leader
Thoughts: own_intention "I need to get to [that side] of the counter." needs_video / wish_for_partner "I want the helper to make room for me at [that side] of the counter." needs_video

Line: leader mic, "Don't put anything in front of the camera, he said, no?"
Disposition: drop, recording_talk.

Return JSON that follows the provided schema exactly."""

OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["groups"],
    "properties": {"groups": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["utterance_ids", "speaker", "speaker_confidence", "speaker_evidence", "disposition",
                     "drop_reason", "thoughts"],
        "properties": {
            "utterance_ids": {"type": "array", "items": {"type": "string"}},
            "speaker": {"type": "string", "enum": ["leader", "helper", "unknown"]},
            "speaker_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "speaker_evidence": {"type": "string"},
            "disposition": {"type": "string", "enum": ["convert", "drop"]},
            "drop_reason": {"type": "string", "enum": ["none", "recording_talk", "filler", "unintelligible", "off_task"]},
            "thoughts": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["kind", "text", "about_person", "object", "referent_resolution", "linked_event_id",
                             "confidence"],
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "text": {"type": "string"},
                    "about_person": {"type": "string", "enum": ["leader", "helper", "none"]},
                    "object": {"type": "string"},
                    "referent_resolution": {"type": "string", "enum": RESOLUTION},
                    "linked_event_id": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                }}},
        }}}},
}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unnamed"


# ----------------------------------------------------------------------------- build

def session_lines(root, rid):
    """All non-empty transcript segments from both mics, with stable ids L0000 / H0000."""
    lines = []
    for mic, prefix in (("leader", "L"), ("helper", "H")):
        segs = load(root / "recordings" / rid / "transcripts" / f"{mic}_trimmed_sync_transcript.json")["segments"]
        for i, s in enumerate(segs):
            text = (s.get("text") or "").strip()
            if text:
                lines.append({"id": f"{prefix}{i:04d}", "mic": mic, "start": round(s["start"], 3),
                              "end": round(s["end"], 3), "speaker": s.get("speaker"), "text": text})
    return sorted(lines, key=lambda x: (x["start"], x["id"]))


def describe_event(e, actions):
    a = next((x for x in actions if x["action_id"] == e.get("response_action_id")), None)
    cue = "+".join(e["cue_channels"])
    if e["type"] == "handover" and a:
        who = "asked for by the receiver" if "receiver initiated" in e["trigger"]["description"] else "offered by the sender"
        return f"{e['start_time']:.1f}-{e['end_time']:.1f} s  {e['actor']} passes {a['noun']} to {e['partner']} ({who}; cue: {cue})"
    if a:
        return f"{a['start_time']:.1f} s  {a['person_id']} {a['verb']} {a['noun']} (cue from {e['partner']}: {cue})"
    return f"{e['start_time']:.1f} s  {e['type']}"


def fmt(line, ctx=False):
    return (f"{'CTX  ' if ctx else ''}{line['id']}  [{line['start']:.1f}-{line['end']:.1f}]  mic={line['mic']}  "
            f"{line['speaker'] or 'no_label'}  \"{line['text']}\"")


def build(args):
    root, seeds, run = Path(args.comind), Path(args.seeds), Path(args.run)
    rids = args.sessions.split(",") if args.sessions else sorted(p.stem for p in seeds.glob("*.json"))
    n_req, req_path = 0, run / "requests.jsonl"
    run.mkdir(parents=True, exist_ok=True)
    with req_path.open("w", encoding="utf-8") as out:
        for rid in rids:
            seed = load(seeds / f"{rid}.json")
            lines = session_lines(root, rid)
            labels = {m: Counter(l["speaker"] or "no_label" for l in lines if l["mic"] == m) for m in ("leader", "helper")}
            label_txt = "; ".join(f"{m} mic: " + ", ".join(f"{k} {v}" for k, v in c.most_common()) for m, c in labels.items())
            end = max((l["end"] for l in lines), default=0.0)
            chunks, w0, idx = [], 0.0, 0
            while w0 < end:
                w1 = w0 + args.window
                target = [l for l in lines if w0 <= l["start"] < w1]
                if target:
                    ctx = [l for l in lines if w0 - args.context <= l["start"] < w0]
                    evs = [e for e in seed["coordination_events"] if w0 - 5 <= e["start_time"] <= w1 + 15]
                    user = "\n".join(
                        ["Session context",
                         "- Roles: the leader knows the recipe and gives instructions; the helper does not know the recipe.",
                         f"- Speaker-label counts for the whole session: {label_txt}",
                         f"- This chunk covers {w0:.0f} to {w1:.0f} s.", "",
                         "CoMind events near this chunk (real actions, for resolving objects and linking):"]
                        + ([f"- {e['event_id']}  {describe_event(e, seed['actions'])}" for e in evs] or ["- none"])
                        + ["", "Context lines (do not group or convert):"]
                        + ([fmt(l, ctx=True) for l in ctx] or ["(none)"])
                        + ["", "Target lines:"] + [fmt(l) for l in target])
                    cid = f"{rid}_c{idx:03d}"
                    params = {"model": MODEL, "max_tokens": args.max_tokens,
                              "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                              "messages": [{"role": "user", "content": user}],
                              "output_config": {"effort": args.effort,
                                                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}}
                    out.write(json.dumps({"custom_id": cid, "params": params}, ensure_ascii=False) + "\n")
                    chunks.append({"custom_id": cid, "window": [w0, w1], "target_ids": [l["id"] for l in target],
                                   "event_ids": [e["event_id"] for e in evs]})
                    n_req += 1
                    idx += 1
                w0 = w1
            save(run / "chunks" / f"{rid}.json", {"recording_id": rid, "lines": lines, "chunks": chunks})
    save(run / "run.json", {"model": MODEL, "effort": args.effort, "window_s": args.window, "context_s": args.context,
                            "max_tokens": args.max_tokens, "sessions": rids, "requests": n_req})
    print(f"built {n_req} requests for {len(rids)} sessions -> {req_path}")


def read_requests(run):
    with (Path(run) / "requests.jsonl").open(encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# ----------------------------------------------------------------------------- estimate

def estimate(args):
    reqs = read_requests(args.run)
    if args.api:
        import anthropic
        client = anthropic.Anthropic()
        sample = random.Random(0).sample(reqs, min(args.sample, len(reqs)))
        per = [client.messages.count_tokens(model=MODEL, system=r["params"]["system"],
                                            messages=r["params"]["messages"]).input_tokens for r in sample]
        in_tok = sum(per) / len(per) * len(reqs)
        how = f"token-counting endpoint on {len(sample)} sampled requests"
    else:
        chars = sum(len(r["params"]["system"][0]["text"]) + len(r["params"]["messages"][0]["content"]) for r in reqs)
        in_tok = chars / 3.5
        how = "local character count (rough, about 3.5 characters per token)"
    lines = sum(len(c["target_ids"]) for p in (Path(args.run) / "chunks").glob("*.json") for c in load(p)["chunks"])
    lo, hi = lines * 60, lines * 200                       # output incl. thinking: assumed range per target line
    cost = lambda o: (in_tok * PRICE_IN + o * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
    print(f"{len(reqs)} requests, {lines} target lines; input tokens about {in_tok:,.0f} ({how})")
    print(f"output tokens assumed {lo:,} to {hi:,} (60 to 200 per line, thinking included)")
    print(f"batch cost estimate, before any cache savings: ${cost(lo):.2f} to ${cost(hi):.2f}")
    print("measure real output on a pilot before trusting the upper end")


# ----------------------------------------------------------------------------- calling the API

def parse_message(msg):
    rec = {"stop_reason": msg.stop_reason, "model": msg.model,
           "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
                     "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                     "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0}}
    if msg.stop_reason == "refusal":
        sd = getattr(msg, "stop_details", None)
        rec.update(status="refused", category=getattr(sd, "category", None))
        return rec
    if msg.stop_reason == "max_tokens":
        rec.update(status="truncated")
        return rec
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        rec.update(status="ok", output=json.loads(text))
    except json.JSONDecodeError as e:
        rec.update(status="bad_json", error=str(e))
    return rec


def append_results(run, records):
    with (Path(run) / "results.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pilot(args):
    import anthropic
    client = anthropic.Anthropic()
    reqs = read_requests(args.run)
    if args.custom_ids:
        want = set(args.custom_ids.split(","))
        reqs = [r for r in reqs if r["custom_id"] in want]
    else:
        reqs = [r for r in reqs if r["custom_id"].startswith(args.session)][: args.chunks]
    for r in reqs:
        try:
            # Synchronous call, so the server-side refusal fallback can be used (it is rejected on batches).
            msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                              **r["params"])
            rec = {"custom_id": r["custom_id"], "mode": "pilot", **parse_message(msg)}
        except anthropic.BadRequestError as e:
            rec = {"custom_id": r["custom_id"], "mode": "pilot", "status": "bad_request", "error": e.message}
        except anthropic.APIStatusError as e:
            rec = {"custom_id": r["custom_id"], "mode": "pilot", "status": f"http_{e.status_code}", "error": e.message}
        append_results(args.run, [rec])
        u = rec.get("usage", {})
        print(f"{r['custom_id']}: {rec['status']}  in {u.get('input')} out {u.get('output')} "
              f"cache_read {u.get('cache_read')}")


def submit(args):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    reqs = read_requests(args.run)
    if args.sessions:
        keep = tuple(args.sessions.split(","))
        reqs = [r for r in reqs if r["custom_id"].startswith(keep)]
    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"])) for r in reqs])
    save(Path(args.run) / "batch.json", {"batch_id": batch.id, "requests": len(reqs), "submitted": time.time()})
    print(f"submitted batch {batch.id} with {len(reqs)} requests")


def collect(args):
    import anthropic
    client = anthropic.Anthropic()
    bid = load(Path(args.run) / "batch.json")["batch_id"]
    while True:
        b = client.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        print(f"{b.processing_status}: {b.request_counts.processing} processing, {b.request_counts.succeeded} done")
        time.sleep(args.poll)
    records = []
    for res in client.messages.batches.results(bid):        # results arrive in any order: key by custom_id
        if res.result.type == "succeeded":
            records.append({"custom_id": res.custom_id, "mode": "batch", **parse_message(res.result.message)})
        else:
            err = getattr(res.result, "error", None)
            records.append({"custom_id": res.custom_id, "mode": "batch", "status": res.result.type,
                            "error": str(getattr(err, "type", err))})
    append_results(args.run, records)
    print(Counter(r["status"] for r in records))


# ----------------------------------------------------------------------------- in-session work

def show(args):
    """Print the prompt for chunks, for a person or an interactive Claude session to answer."""
    import io, sys
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    for r in read_requests(args.run):
        if r["custom_id"].startswith(args.prefix):
            out.write(f"\n===== {r['custom_id']}\n{r['params']['messages'][0]['content']}\n")
    out.flush()


def ingest(args):
    """Bring in answers written outside the API (one <custom_id>.json per chunk, each matching
    OUTPUT_SCHEMA), after checking them against the schema the API would have enforced."""
    import jsonschema
    validator = jsonschema.Draft202012Validator(OUTPUT_SCHEMA)
    known = {r["custom_id"] for r in read_requests(args.run)}
    records, bad = [], 0
    for f in sorted(Path(args.dir).glob("*.json")):
        cid = f.stem
        if cid not in known:
            print(f"skip {f.name}: not a chunk in this run")
            continue
        obj = load(f)
        errs = list(validator.iter_errors(obj))
        if errs:
            bad += 1
            print(f"{cid}: {len(errs)} schema errors, first: {errs[0].message}")
            continue
        records.append({"custom_id": cid, "mode": "in_session", "status": "ok", "model": args.model, "output": obj})
    append_results(args.run, records)
    print(f"ingested {len(records)} chunks, rejected {bad}")


# ----------------------------------------------------------------------------- merge

def latest_results(run):
    best = {}
    path = Path(run) / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["custom_id"] not in best or r["status"] == "ok":
                best[r["custom_id"]] = r
    return best


def merge(args):
    run, seeds, out = Path(args.run), Path(args.seeds), Path(args.out)
    results = latest_results(run)
    meta = load(run / "run.json")
    report, review = Counter(), []
    only = set(args.sessions.split(",")) if args.sessions else None
    for cpath in sorted((run / "chunks").glob("*.json")):
        info = load(cpath)
        rid = info["recording_id"]
        if only and rid not in only:
            continue
        lines = {l["id"]: l for l in info["lines"]}
        doc = load(seeds / f"{rid}.json")
        doc["schema_version"] = "0.2.0"
        pass_id = "p_thoughts"
        doc["annotation_passes"] = [p for p in doc["annotation_passes"] if p["pass_id"] != pass_id] + [
            {"pass_id": pass_id, "layer": "thoughts", "annotator_id": MODEL, "audio": "on",
             "tool": "scripts/speech_to_thoughts.py",
             "notes": f"run {run.name}, effort {meta['effort']}; reads transcripts only, no audio or video. Review a sample by hand."}]
        resources = {r["resource_id"]: r for r in doc["resources"]}
        events = {e["event_id"]: e for e in doc["coordination_events"]}
        actions = {a["action_id"]: a for a in doc["actions"]}
        thoughts = []
        for ch in info["chunks"]:
            res = results.get(ch["custom_id"])
            if not res or res["status"] != "ok":
                report[f"chunk_{res['status'] if res else 'missing'}"] += 1
                continue
            targets, seen = set(ch["target_ids"]), Counter()
            for gi, g in enumerate(res["output"]["groups"]):
                ids = [i for i in g["utterance_ids"] if i in targets]
                report["unknown_ids"] += len(g["utterance_ids"]) - len(ids)
                seen.update(ids)
                if not ids or g["disposition"] == "drop":
                    report[f"dropped_{g['drop_reason']}"] += 1
                    continue
                if g["speaker"] == "unknown":
                    report["skipped_unknown_speaker"] += 1     # a thought needs an owner
                    continue
                owner = g["speaker"]
                utts = [{"text": lines[i]["text"], "start_time": lines[i]["start"], "end_time": lines[i]["end"],
                         "mic": lines[i]["mic"], "whisper_speaker": lines[i]["speaker"], "speaker_id": owner,
                         "speaker_attribution": "llm_inferred"} for i in sorted(ids, key=lambda i: lines[i]["start"])]
                for ti, t in enumerate(g["thoughts"]):
                    tid = f"t{ch['custom_id'][-3:]}_{gi:03d}_{ti}"
                    obj = slug(t["object"]) if t["object"].strip() else None
                    if obj and obj not in resources:
                        resources[obj] = {"resource_id": obj, "type": "object", "subtype": "",
                                          "name": t["object"].strip(), "comind_instance": None}
                    led, fulfilled = [], "unknown"
                    ev = events.get(t["linked_event_id"]) if t["linked_event_id"] else None
                    if t["linked_event_id"] and not ev:
                        report["bad_link"] += 1
                    if ev and ev.get("response_action_id") in actions:
                        act = actions[ev["response_action_id"]]
                        led, fulfilled = [act["action_id"]], "yes"
                        if act["person_id"] == owner and t["kind"] in ("own_intention", "joint_intention"):
                            act["intention_thought_id"] = tid
                        elif act["person_id"] != owner and act.get("influence") and act["influence"]["partner"] == owner:
                            act["influence"]["trigger"]["thought_id"] = tid
                            if ev["trigger"].get("person_id") == owner:
                                ev["trigger"]["thought_id"] = tid
                    about = None if t["about_person"] == "none" else t["about_person"]
                    thoughts.append({
                        "thought_id": tid, "person_id": owner, "known_to": [owner], "time": utts[0]["start_time"],
                        "kind": t["kind"], "text": t["text"], "about_person": about, "object_resource_id": obj,
                        "referent_resolution": t["referent_resolution"],
                        "source": {"type": "speech", "utterances": utts, "speaker_attribution": "llm_inferred"},
                        "led_to_action_ids": led, "fulfilled": fulfilled, "pass_id": pass_id,
                        "confidence": min(t["confidence"], g["speaker_confidence"], key=["low", "medium", "high"].index),
                        "notes": f"speaker: {g['speaker_evidence']}"})
                    report[f"kind_{t['kind']}"] += 1
                    review.append((rid, utts, owner, g["speaker_confidence"], t))
            report["lines_not_covered"] += len(targets - set(seen))
            report["lines_in_two_groups"] += sum(1 for v in seen.values() if v > 1)
        doc["resources"] = sorted(resources.values(), key=lambda r: r["resource_id"])
        doc["thoughts"] = sorted(thoughts, key=lambda t: (t["time"], t["thought_id"]))
        doc["notes"] = (doc.get("notes", "") + f" Thoughts added by {MODEL} (run {run.name}); speakers are model-inferred.").strip()
        save(out / f"{rid}.json", doc)
    sample = random.Random(0).sample(review, min(args.review, len(review)))
    md = ["# Thought review sample", "", f"Run `{run.name}`, {len(sample)} of {len(review)} thoughts, chosen at random.", ""]
    for rid, utts, owner, sconf, t in sample:
        md += [f"**{rid[:8]} @ {utts[0]['start_time']:.1f} s**: " + " / ".join(f"{u['mic']} mic: \"{u['text']}\"" for u in utts),
               f"- speaker: {owner} ({sconf})", f"- {t['kind']}: *{t['text']}* ({t['referent_resolution']})", ""]
    (run / "review_sample.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(dict(sorted(report.items())), indent=1))
    print(f"wrote {out} and {run / 'review_sample.md'}; now run validate_labels.py on {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--comind", required=True); b.add_argument("--seeds", required=True); b.add_argument("--run", required=True)
    b.add_argument("--sessions", help="comma-separated recording ids; default all seeds")
    b.add_argument("--window", type=float, default=120.0, help="seconds of dialogue per request")
    b.add_argument("--context", type=float, default=30.0, help="seconds of earlier dialogue shown as context")
    b.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    b.add_argument("--max-tokens", type=int, default=16000)
    e = sub.add_parser("estimate"); e.add_argument("--run", required=True)
    e.add_argument("--api", action="store_true"); e.add_argument("--sample", type=int, default=20)
    p = sub.add_parser("pilot"); p.add_argument("--run", required=True)
    p.add_argument("--session", default=""); p.add_argument("--chunks", type=int, default=3)
    p.add_argument("--custom-ids", help="rerun specific chunks, e.g. refused ones")
    s = sub.add_parser("submit"); s.add_argument("--run", required=True); s.add_argument("--sessions")
    c = sub.add_parser("collect"); c.add_argument("--run", required=True); c.add_argument("--poll", type=int, default=60)
    m = sub.add_parser("merge"); m.add_argument("--run", required=True); m.add_argument("--seeds", required=True)
    m.add_argument("--out", required=True); m.add_argument("--review", type=int, default=40)
    m.add_argument("--sessions", help="comma-separated recording ids; default all")
    sh = sub.add_parser("show"); sh.add_argument("--run", required=True); sh.add_argument("--prefix", required=True)
    ig = sub.add_parser("ingest"); ig.add_argument("--run", required=True); ig.add_argument("--dir", required=True)
    ig.add_argument("--model", default=f"{MODEL} (interactive session)")
    args = ap.parse_args()
    {"build": build, "estimate": estimate, "pilot": pilot, "submit": submit, "collect": collect, "merge": merge,
     "show": show, "ingest": ingest}[args.cmd](args)


if __name__ == "__main__":
    main()
