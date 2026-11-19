# Reading a Partner's Intent from What Is Visible

CS 7170 (Multimodal Video Understanding, Northeastern, Fall 2026) semester project by Angshuman Basu and Zilu (Max).

Two people cook together. The leader has the recipe and the helper does not. **Can a model predict the leader's next instruction from muted egocentric video of both people plus the recipe, before the instruction is spoken?** The data is [CoMind](https://comind.ethz.ch), 55 training sessions of paired head-mounted recordings with transcripts.

## The test

Each leader instruction is labelled as a (step, action, object) triple against a recipe task graph. For each instruction the model gets:

- the dish's recipe graph,
- the minutes elapsed,
- the last 32 s of both egocentric views at 1 fps, ending 1 s before the leader starts speaking (no audio).

It then answers three five-option questions: which recipe step, which kind of action (16 classes), which object. The three are scored separately and jointly. Splits are by session (dev 118 items, test 157).

## Pipeline

| stage | script | output |
|---|---|---|
| CoMind annotations to seed labels | `scripts/comind_to_labels.py` | `labels/seed/` (local only) |
| speech to leader and helper thoughts | `scripts/speech_to_thoughts.py`, `scripts/tag_heard_dependency.py` | `labels/thoughts/` (local only) |
| session split | `scripts/make_split.py` | `splits/thoughts_v1.json` |
| generic recipe graph per dish | written from the dish name only; checked by `scripts/validate_recipes.py` | `labels/recipes/graphs/` |
| instruction to (step, action, object) | `scripts/instruction_map.py` | `labels/recipes/instruction_map/` (local only) |
| multiple-choice items | `scripts/build_mc_items.py` | `runs/mc_v2/items.jsonl` |
| no-video baselines | `scripts/baselines_novideo.py` | `runs/mc_v2/baselines.txt` |
| egocentric video download | `scripts/fetch_ego_videos.py` | 110 MP4s, 183.6 GiB |
| frames before each cutoff | `scripts/extract_frames.py` | `runs/mc_v2/frames_w32/` |
| blind and zero-shot video scoring | `scripts/blind_llm.py` (`--frames` for video) | `runs/mc_v2/{blind,video_w32}/` |
| cluster env and jobs (Explorer) | `scripts/cluster/` | |

The recipe graphs are generic: one agent identified each dish from the transcripts, and a second agent that never saw the recordings wrote the recipe from the dish name only, so recipes carry no session-specific information. Each step has a kind, its true prerequisites and a visible checkpoint.

## What is not in this repo

CoMind video, annotations and transcripts are not redistributed here; get them from the dataset authors. Labels derived from the transcripts (seed labels, thoughts, dish evidence, instruction map) quote participants' speech, so they are kept out of this public repo (see `.gitignore`). The scripts above regenerate them from a local CoMind copy.

## Results so far (test split, chance 20% per question)

| model | step | action | object | all three |
|---|---|---|---|---|
| answer frequency prior | 20.0% | 28.0% | 18.7% | 1.3% |
| Qwen3-VL-4B, recipe only (blind) | 31.9% | 24.8% | 34.2% | 1.9% |
| Qwen3-VL-4B, recipe + muted video (zero-shot) | running | | | |

n = 135 to 157 per question, so the 95% margin is about 8 points. All labels are model-made and not yet hand-checked.

## Docs

- [docs/proposal.md](docs/proposal.md) and [docs/proposal/](docs/proposal/): the project proposal
- [docs/project-summary.md](docs/project-summary.md): short summary
- [docs/literature-survey-multiagent.md](docs/literature-survey-multiagent.md): related work
- [docs/label-schema-implicit-coordination.md](docs/label-schema-implicit-coordination.md): label schema; machine-checkable with [schemas/coordination_labels.schema.json](schemas/coordination_labels.schema.json) and `scripts/validate_labels.py`
