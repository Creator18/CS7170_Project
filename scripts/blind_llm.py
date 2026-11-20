"""Blind and zero-shot video baselines on mc_v2 items with a small open model.

Without --frames the model sees the recipe and the options only (blind). With --frames DIR it
also sees two muted clips, the leader's and the helper's egocentric views, ending just before the
leader starts speaking (frames from scripts/extract_frames.py). Everything else is identical.

Each question (step, action, object) is a separate prompt with the dish name, the recipe steps,
the minutes elapsed at the cutoff, and options A to E. The answer is read from the next-token
probabilities of the letters A to E. Every question is scored under all 5 cyclic rotations of
the option order and the per-option log-probabilities are summed, which removes the model's
letter-position bias. No session history is given (no earlier instructions, no state).

The same model (Qwen3-VL-4B-Instruct) is planned for the zero-shot video baseline, so the
blind and video runs differ only in the frames.

    python scripts/blind_llm.py --items runs/mc_v2/items.jsonl --out runs/mc_v2/blind --dry-run 2
    python scripts/blind_llm.py --items items.jsonl --graphs labels/recipes/graphs --out OUT --model Qwen/Qwen3-VL-4B-Instruct
    python scripts/blind_llm.py --items items.jsonl --frames frames_w32 --out OUT --batch 2
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

LETTERS = "ABCDE"
ACTION_TEXT = {
    "fetch_or_hand": "fetch something or hand it over", "wash": "wash something", "peel": "peel something",
    "cut": "cut, chop or slice something", "grate": "grate something", "open_or_crack": "open a package or crack an egg",
    "add": "add or put something into or onto the food", "mix": "mix, stir or knead", "heat_control": "adjust the stove, oven or heat",
    "transfer": "move food to a plate or pot, drain or serve", "wait_or_stop": "wait, stop or be careful",
    "taste_or_check": "taste or check something", "clean_or_tidy": "clean or tidy up", "cook": "cook, fry, boil or bake a food",
    "cover_or_uncover": "put on or take off a lid or cover", "other": "something else"}
QUESTION = {
    "step": "Which recipe step will this instruction be about?",
    "action": "What kind of action will the leader ask the helper to do?",
    "object": "Which object or ingredient will the instruction be about?"}


def prompt(item, field, opts, steps, clip=None):
    recipe = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(steps.values()))
    shown = {"step": lambda o: steps[o]["text"], "action": lambda o: ACTION_TEXT[o], "object": lambda o: o}[field]
    lines = [f"Two people are cooking {item['dish_name']}. The leader has the recipe below; the helper does not.",
             f"The recipe:\n{recipe}",
             f"It is {item['cutoff_time'] / 60:.1f} minutes into the session. The leader is about to give the helper an instruction."
             + (f" The two silent clips above show the last {clip:.0f} seconds from the leader's and the helper's"
                " head-mounted cameras, ending just before the leader speaks." if clip is not None else ""),
             QUESTION[field]]
    lines += [f"{LETTERS[i]}. {shown(o)}" for i, o in enumerate(opts)]
    lines.append("Answer with a single letter.")
    return "\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--graphs", default="labels/recipes/graphs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", type=int, default=0, help="print N prompts and exit (no model)")
    ap.add_argument("--frames", default="", help="frame dir from extract_frames.py; empty = blind")
    ap.add_argument("--px", type=int, default=320, help="frame side in pixels fed to the model (multiple of 32)")
    a = ap.parse_args()
    graphs = {}
    for p in Path(a.graphs).glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        graphs[d["dish_id"]] = {s["step_id"]: s for s in d["steps"]}
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    items = [it for it in items if it["split"] in a.splits.split(",")]
    if a.limit:
        items = items[: a.limit]
    clips = {}
    if a.frames:
        for l in open(Path(a.frames) / "index.jsonl", encoding="utf-8"):
            r = json.loads(l)
            clips[r["item_id"]] = r
        items = [it for it in items if it["item_id"] in clips]
    jobs = []  # (item index, field, rotation, prompt, rotated options)
    for k, it in enumerate(items):
        clip = None
        if a.frames:
            clip = max(1.0, clips[it["item_id"]]["window"] + 1.0)
        for field, q in it["questions"].items():
            for r in range(5):
                opts = q["options"][r:] + q["options"][:r]
                jobs.append((k, field, r, prompt(it, field, opts, graphs[it["dish_id"]], clip), opts))
    if a.dry_run:
        for j in jobs[: a.dry_run * 5: 5]:
            print(j[3], "\n" + "-" * 60)
        print(f"{len(items)} items, {len(jobs)} prompts")
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()
    letter_ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(t) == 1 for t in letter_ids), letter_ids
    letter_ids = [t[0] for t in letter_ids]
    print("gpu", torch.cuda.get_device_name(0), "| prompts", len(jobs), flush=True)
    video_cache = {}

    def load_clip(item_id):
        if item_id not in video_cache:
            from PIL import Image
            from transformers.video_utils import VideoMetadata
            n = len(clips[item_id]["times"])
            vids = []
            for v in ("leader", "helper"):
                fr = [Image.open(Path(a.frames) / item_id / f"{v}_{i:02d}.jpg").convert("RGB").resize((a.px, a.px))
                      for i in range(n)]
                vids.append(fr)
            meta = VideoMetadata(total_num_frames=n, fps=1.0, width=a.px, height=a.px, frames_indices=list(range(n)))
            video_cache.clear()
            video_cache[item_id] = (vids, meta)
        return video_cache[item_id]

    def content(j):
        if not a.frames:
            return [{"type": "text", "text": j[3]}]
        return [{"type": "text", "text": "Leader's view:"}, {"type": "video"},
                {"type": "text", "text": "Helper's view:"}, {"type": "video"},
                {"type": "text", "text": j[3]}]

    # Checkpoint: every scored prompt is appended to partial.jsonl as soon as its batch finishes,
    # so a job that hits its time limit can be resubmitted with the same arguments and resumes.
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha256(json.dumps([[j[0], j[1], j[2], j[3]] for j in jobs]).encode()).hexdigest()[:16]
    sig += f"|{a.model}|{a.frames}|{a.px}"
    part = out / "partial.jsonl"
    done = {}
    if part.exists():
        with open(part, encoding="utf-8") as f:
            head = json.loads(f.readline())
            assert head.get("sig") == sig, f"{part} was written by a different run ({head.get('sig')} vs {sig}); use a new --out"
            for l in f:
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    break  # a line cut off by the kill
                done[r["j"]] = r["lp"]
        print(f"resuming: {len(done)}/{len(jobs)} prompts already scored", flush=True)
    else:
        part.write_text(json.dumps({"sig": sig}) + "\n")
    pf = open(part, "a", encoding="utf-8")

    scores = collections.defaultdict(lambda: collections.defaultdict(float))
    nan = 0

    def add(j, lp):
        nonlocal nan
        if lp is None:
            nan += 1
            return
        k, field, r, _, opts = jobs[j]
        for i, o in enumerate(opts):
            scores[(k, field)][o] += lp[i]

    for j, lp in done.items():
        add(j, lp)
    todo = [i for i in range(len(jobs)) if i not in done]
    for b in range(0, len(todo), a.batch):
        idx = todo[b: b + a.batch]
        chunk = [jobs[i] for i in idx]
        texts = [proc.apply_chat_template([{"role": "user", "content": content(j)}],
                                          tokenize=False, add_generation_prompt=True) for j in chunk]
        if a.frames:
            vids, metas = [], []
            for j in chunk:
                v, m = load_clip(items[j[0]]["item_id"])
                vids += v
                metas += [m, m]
            n = len(vids[0])
            enc = proc(text=texts, videos=vids, video_metadata=metas, do_sample_frames=False,
                       size={"shortest_edge": 32 * 32, "longest_edge": n * a.px * a.px},
                       return_tensors="pt", padding=True).to("cuda")
            if b == 0:
                print("tokens per prompt", enc["input_ids"].shape[1], "video grid", enc["video_grid_thw"][0].tolist(), flush=True)
        else:
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits[:, letter_ids], dim=-1).cpu()
        for i, row in zip(idx, lp):
            v = None if torch.isnan(row).any() else [round(x, 5) for x in row.tolist()]
            add(i, v)
            pf.write(json.dumps({"j": i, "lp": v}) + "\n")
        pf.flush()
        if b // a.batch % 50 == 0:
            print(f"  {len(done) + b + len(chunk)}/{len(jobs)}", flush=True)
    pf.close()
    acc = collections.defaultdict(collections.Counter)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for k, it in enumerate(items):
            joint, rec = True, {"item_id": it["item_id"], "split": it["split"]}
            for field, q in it["questions"].items():
                s = scores[(k, field)]
                pred = max(q["options"], key=lambda o: s.get(o, float("-inf")))
                ok = pred == q["options"][q["answer_index"]]
                joint &= ok
                acc[(it["split"], field)]["ok"] += ok
                acc[(it["split"], field)]["n"] += 1
                rec[field] = {"pred": pred, "correct": ok, "scores": {str(o): round(v, 3) for o, v in s.items()}}
            acc[(it["split"], "joint")]["ok"] += joint
            acc[(it["split"], "joint")]["n"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lines = [f"model {a.model}, fp16, 5 rotations per question, NaN rows skipped: {nan}",
             f"input: {'blind (no video)' if not a.frames else f'video {a.frames}, {a.px}px, leader+helper views'}"]
    for sp in a.splits.split(","):
        for field in ("step", "action", "object", "joint"):
            c = acc[(sp, field)]
            if c["n"]:
                lines.append(f"{sp:5} {field:7} n={c['n']:4} acc={c['ok'] / c['n']:.1%}")
    (out / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
