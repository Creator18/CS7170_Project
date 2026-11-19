"""Check generic recipe task graphs in labels/recipes/graphs/.

Each file is one dish:
{
  "version": "0.1.0",
  "dish_id": "slug", "dish_name": "...", "components": ["..."],
  "source": "how the recipe was written",
  "ingredients": [{"id": "i_x", "name": "...", "amount": "..." or null}],
  "equipment":   [{"id": "e_x", "name": "..."}],
  "steps": [{"step_id": "s1", "component": "...", "text": "...",
             "kind": "fetch|prep|cook|combine|serve|clean",
             "ingredients": ["i_x"], "equipment": ["e_x"],
             "requires": ["s0"], "checkpoint": "what is true when the step is done"}]
}
`requires` are the step's prerequisites, so the steps form a task graph. Checks:
ids unique and referenced ids exist, every step reachable, the graph is acyclic,
no em dashes. Also checks that every session in dishes.json maps to a graph.

    python scripts/validate_recipes.py labels/recipes
"""
import argparse
import json
import sys
from pathlib import Path

KINDS = {"fetch", "prep", "cook", "combine", "serve", "clean"}
STEP_KEYS = {"step_id", "component", "text", "kind", "ingredients", "equipment", "requires", "checkpoint"}


def check_graph(d):
    errs = []
    for k in ("dish_id", "dish_name", "components", "source", "ingredients", "equipment", "steps"):
        if k not in d:
            errs.append(f"missing {k}")
    if errs:
        return errs
    ing = {i["id"] for i in d["ingredients"]}
    eq = {e["id"] for e in d["equipment"]}
    if len(ing) != len(d["ingredients"]) or len(eq) != len(d["equipment"]):
        errs.append("duplicate ingredient or equipment id")
    steps = {}
    for s in d["steps"]:
        sid = s.get("step_id")
        if sid in steps:
            errs.append(f"duplicate step {sid}")
        steps[sid] = s
        missing = STEP_KEYS - s.keys()
        if missing:
            errs.append(f"{sid}: missing {sorted(missing)}")
            continue
        if s["kind"] not in KINDS:
            errs.append(f"{sid}: bad kind {s['kind']}")
        if s["component"] not in d["components"]:
            errs.append(f"{sid}: component {s['component']!r} not in components")
        errs += [f"{sid}: unknown ingredient {i}" for i in s["ingredients"] if i not in ing]
        errs += [f"{sid}: unknown equipment {e}" for e in s["equipment"] if e not in eq]
    for sid, s in steps.items():
        errs += [f"{sid}: unknown prerequisite {r}" for r in s.get("requires", []) if r not in steps]
    # cycle check (DFS with colours)
    state = {}

    def visit(n, path):
        if state.get(n) == 1:
            errs.append("cycle: " + " -> ".join(path + [n]))
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for r in steps[n].get("requires", []):
            if r in steps:
                visit(r, path + [n])
        state[n] = 2

    for n in steps:
        visit(n, [])
    used = {i for s in d["steps"] for i in s.get("ingredients", [])}
    errs += [f"ingredient {i} used by no step" for i in sorted(ing - used)]
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="labels/recipes")
    root = Path(ap.parse_args().root)
    bad = 0
    graphs = {}
    for p in sorted((root / "graphs").glob("*.json")):
        text = p.read_text(encoding="utf-8")
        d = json.loads(text)
        errs = check_graph(d)
        if "—" in text:
            errs.append("contains an em dash")
        if d.get("dish_id") != p.stem:
            errs.append(f"dish_id {d.get('dish_id')} != file name")
        graphs[p.stem] = d
        if errs:
            bad += 1
            print(f"FAIL {p.name}")
            for e in errs:
                print("   ", e)
    dishes = json.loads((root / "dishes.json").read_text(encoding="utf-8"))
    for s in dishes["sessions"]:
        if s.get("dish_id") not in graphs:
            bad += 1
            print(f"FAIL session {s['recording_id'][:8]}: no graph for dish_id {s.get('dish_id')}")
    print(f"{len(graphs)} graphs, {len(dishes['sessions'])} sessions, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
