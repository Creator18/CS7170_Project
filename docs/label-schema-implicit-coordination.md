# Label Schema: Task Stage and Implicit Coordination (draft v0.2.0)

**Status:** draft for the team to try on one session before committing to it.

**Changes on 2026-10-22 (doc only):** the recipe comes from CoMind's annotations and must be the plan, not the performance; checkpoints are defined per immediate task; and a **sports variant** (section 6d) where points are positive or negative events and checkpoints are windows scaled to the points needed to win, with a soft target of leading, preferably with 75% of the points.

**Changes in v0.3.0 (doc only, not yet in the JSON schema or validator):** the **recipe is given to the model as input**, and a **checkpoint-awareness layer** (section 6c) defines progress as a recipe step that is done *and* known to be done by both people. Each person's awareness is marked by its first observable evidence: said, looked or acted on.

**Changes in v0.2.1:** an optional `heard_dependency` tag on each thought, saying whether it could exist if its owner had never heard the partner (`independent`, or `depends_on_partner_speech` with a reason such as `acknowledges_partner` or `answers_partner`). A model that must not hear the partner gets only the independent thoughts. See [labels/thoughts/README.md](../labels/thoughts/README.md).

**Changes in v0.2.0:** a **thoughts layer**, where everything said aloud is recast as the speaker's private first-person thought (section 6a), and **dense actions**, where every necessary action by either person is labelled with whether the partner influenced it (section 6b). v0.1.1 added `unknown` to `cue_channels`, for CoMind events that record no cue type, so they are never mistaken for implicit ones.

**Seed labels for all 55 CoMind training sessions** are in [labels/seed/](../labels/seed/), converted automatically from CoMind's annotations. They contain the coordination events; the task layer is a placeholder until someone annotates it from video. See that folder's README for exactly what's in them and what isn't.
**Files:**
- [schemas/coordination_labels.schema.json](../schemas/coordination_labels.schema.json): the machine-checkable schema (JSON Schema 2020-12)
- [scripts/validate_labels.py](../scripts/validate_labels.py): schema check plus cross-reference checks
- [schemas/examples/01d63ce5_example.json](../schemas/examples/01d63ce5_example.json): worked example built on real CoMind events

```bash
python scripts/validate_labels.py labels/<recording_id>.json --comind data/comind/v1/annotations
```

---

## 1. What this adds, and what it leaves alone

CoMind labels three kinds of moment: joint attention, an action the helper took because the leader cued it, and handovers. Its cue rule records *that* the leader prompted something, not *why the action made sense at that point in the work*, and nothing is labelled when no one gave a cue.

This schema adds two layers on top:

1. **A task layer**, per person: what goal, step and substep each person is on, over time. This is "we are currently at stage S of task T".
2. **A coordination layer**: events that link the two people, each with the task context on both sides, the cue channel the people actually used, and whether the action depended on the partner.

CoMind's own labels are **linked by reference and never overwritten.** If the leader asked, that stays recorded. The task stage sits alongside it, because both can be true at once: the leader asked *because* it was that stage.

## 2. Design rules

1. **Add, don't replace.** Every CoMind event we build on carries a `comind_ref` back to it.
2. **Label with audio on; record what video alone supports.** The people's cues are judged with sound. Whether an event can be worked out *without* sound is a separate field, best filled by a muted pass. The model can be given muted input; the labels must still know which moments were cued.
3. **A cue is an intentional signal aimed at the partner.** Speech addressed to them, pointing, nodding, holding something out, placing something *for* them. Gaze, visible progress and where someone stands are **not** cues. They are what implicit coordination runs on.
4. **"Implicit" is derived, not chosen.** An event is implicit when `cue_channels` is `["none"]` and `partner_dependency` is `"yes"`. The validator reports it, so annotators never have to pick between "cued" and "implicit" in a separate field that could contradict the cue field.
5. **Dense task layer, dense actions.** Steps and substeps cover the whole session for both people, and so do individual actions (grab salt), including the obvious ones nobody called out. Each action says whether the partner influenced it. This is the most expensive layer to label; see section 7 for how to keep it manageable.
6. **Every label has provenance:** which pass, which annotator, audio on or off, and a categorical confidence.
7. **Everything is checked by machine.** CoMind's own files contain confidence values of 25.0, events with zero or eight boxes, and missing fields. The schema and validator stop that kind of drift.

## 3. Task layer: modelled on Ego4D Goal-Step

Borrowed from [Ego4D Goal-Step](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a65606fa1a6849450550325832036e5-Paper-Datasets_and_Benchmarks.pdf) (Song, Byrne, Nagarajan, Wang, Martin, Torresani, NeurIPS 2023 Datasets and Benchmarks), checked against the paper:

| Goal-Step | Here |
| --- | --- |
| Three levels, goal, step, substep, each with a time interval, category and free-text description | same, `level` and `parent_id` |
| Annotators write "the camera wearer did X in order to Y" and put the intent Y ahead of the movement X | `description` follows the template; `intent` holds Y alone |
| `is_relevant`: essential, optional or irrelevant | `relevance`, same three values |
| `is_continued`, the completion signal | `is_continued` |
| Taxonomy grown from the data: annotators propose missing categories, reviewed periodically | `label` is free snake_case, consolidated in review |
| Steps average 32.5 s | a reasonable target granularity for steps |

Added for two people:

| Field | Meaning |
| --- | --- |
| `person_ids` | who carries out the task; two entries for a step done together. Goals are usually shared |
| `ordinal` | position among siblings under the same parent. Together with the parent and `is_continued`, this is the **stage** |
| `phase` | optional: `prep`, `execute`, `cleanup`, `wait`, `other` |
| `assigned_by` | how the person came to own the task: `self`, `leader`, `partner`, `unclear`. CoMind sessions often open with the leader assigning jobs aloud |

**Reading "current task and stage" for any moment:** take the person's innermost task segment covering that time. Its label is the current task; its `ordinal`, its parent and the goal give the stage. This also makes "what is my immediate task" answerable at *any* time, not only at CoMind's sparse cued moments.

**An action may serve the partner's task.** In the example, the leader passing the salt is linked to the helper's `season_chicken` substep, not the leader's own rice. That is coordination, and the schema allows it deliberately.

## 4. Persons

| Field | Meaning |
| --- | --- |
| `person_id` | stable name used everywhere, e.g. `leader`, `helper` |
| `role` | `leader`, `helper`, or `peer` for future data without roles |
| `knows_plan` | CoMind leaders know the recipe; helpers do not |
| `comind_side` | `left` or `right`, because CoMind's handover file names people only by side |
| `side_mapping_verified` | stays `false` until checked against the side-by-side video. The validator warns while it is false |

## 5. Resources

Every fixture, space and object that appears in an event, each treated as **exclusive**: one person at a time.

| `type` | Examples |
| --- | --- |
| `fixture` | drawer, cupboard, sink, stove, a section of counter |
| `space` | the working area someone stands in, e.g. "that side" |
| `object` | a specific pot, knife or salt shaker; use CoMind's instance label (`strainer 3`) when one exists |

## 6. Coordination events

Three forms:

| `type` | Use for |
| --- | --- |
| `response` | the actor did something because of the partner: cued, or implicit |
| `handover` | an object changes hands. `actor` is the sender. A response action is required |
| `resource_conflict` | both people need the same exclusive resource, or one asks the other for it. A `resource` block is required |

Core fields:

| Field | Meaning |
| --- | --- |
| `actor`, `partner` | whose behaviour the event explains, and the other person |
| `trigger.kind` | what set it off. Cues: `partner_speech`, `partner_gesture`. Not cues: `partner_action_completed`, `partner_moved`, `partner_state`, `partner_gaze`, `resource_freed`, `object_staged`, `partner_need_inferred`. Also `own_plan` and `unclear` |
| `cue_channels` | intentional signals the people used: any of `speech`, `gesture`, or exactly `["none"]` |
| `partner_dependency` | CoMind's own counterfactual test: would the actor have done this without the partner? `yes`, `no`, `unclear` |
| `task_context` | the actor's task, the partner's task and the shared goal at that moment |
| `speech` | the utterances involved, with which microphone, WhisperX's label, who actually spoke, and how that was decided |
| `resource` | for conflicts: overlap, outcome (`one_yields`, `both_wait`, `reroute`, `sequential`, `contact`, `unresolved`, `unknown`), who yielded, wait time, how it was resolved, and `real` or `composite` |
| `related_event_ids` | other events describing the same moment, e.g. a handover and the cued receive inside it |
| `visually_resolvable` | could someone who can't hear work this out? `yes`, `partial`, `no`, `not_assessed` |
| `first_person_account` | reserved for participants' own accounts. Always `null` for CoMind |
| `computed` | filled by scripts only, never by hand: partner in view, distance, speech in the preceding window |

### How the cue fields combine

| `cue_channels` | `partner_dependency` | Meaning |
| --- | --- | --- |
| `speech` and/or `gesture` | `yes` | cued coordination |
| `none` | `yes` | **implicit coordination** |
| `none` | `no` | not a coordination event: don't label it as a `response` (the schema rejects it) |
| any | `unclear` | flag for review |

### CoMind's edge cases, under these rules

| Situation (from CoMind's annotator guide or data) | CoMind's tag | Here |
| --- | --- | --- |
| Helper turns on the stove after the leader puts a pan on | implicit | `partner_action_completed`, no cue: **implicit** |
| Helper means to hand over salt, sees the leader's hands are full, puts it down | gestural | `partner_state`, no cue: **implicit** |
| Leader places spices near the helper, helper picks them up | gestural | **cued** (gesture) only if placed *for* the helper. If the leader just put them down while working: `object_staged`, no cue, **implicit** |
| "Sorry, I need to get that side." | not labelled | `resource_conflict` on a `space`, cue `speech` |

The first three show why the cue field has to be about **intent to signal**. CoMind puts some partner-reading under "gestural" and some under "implicit"; this schema makes the boundary explicit.

### Speaker attribution

Each person's glasses hear both people, WhisperX labels are anonymous and differ between the two microphones, and the same words get transcribed differently. So every utterance records `mic`, `whisper_speaker`, `speaker_id` and `speaker_attribution`, which is one of `manual`, `auto_loudness` (your own voice is loudest on your own glasses), `inferred_label_dominance` (the label that dominates a person's own microphone is probably them), or `unknown`. Never guess silently.

## 6a. Thoughts: the silent kitchen

The evaluation imagines a kitchen where nobody speaks. Everything CoMind's participants said aloud is recast as what they privately intended, wished, believed or wondered, and nothing tells the model it was ever spoken.

| Said aloud | Becomes | `kind` |
| --- | --- | --- |
| "Can you put pepper on it?" | *I want the helper to put pepper on the chicken.* | `wish_for_partner` |
| "I'm going for the knife." | *I'm going for the knife.* | `own_intention` |
| "Let's plate it now." | *We should plate it now.* | `joint_intention` |
| "I think we need more salt." | *We need more salt.* | `belief` |
| "Should I peel these?" | *I wonder whether I should peel the potatoes.* | `question` |
| "Okay." (accepting a plan) | *I accept that plan.* | `acknowledgement` |
| "Don't put anything in front of the camera." | dropped: about the recording, not the task | none |

**The rule:** a thought is known only to its owner. `known_to` must equal `[person_id]`, and the validator enforces it. A model playing one person is given that person's own thoughts and never the partner's, so the partner's thoughts have to be read from behaviour.

| Field | Meaning |
| --- | --- |
| `person_id`, `known_to` | whose mind; always just that person |
| `time` | the start of the utterance it came from, so the thought existed at least by then |
| `text` | first person, present tense, never phrased as speech; names the partner by role, and the object when known |
| `about_person` | who it concerns, e.g. the partner for a wish |
| `referent_resolution` | how "that" in "pass me that" was filled in: `named_in_speech`, `from_linked_event`, `from_context`, `needs_video` (text keeps a `[placeholder]`), `not_needed` |
| `source` | the transcript segments it came from, from both microphones, and how the speaker was decided |
| `led_to_action_ids`, `fulfilled` | which actions, by either person, carried it out, and whether they did |

**Two things to keep in mind when reading results.** The participants *did* hear each other, so a helper who grabbed the pepper on cue would often not have guessed it in a real silent kitchen; the test measures whether the partner's thought *could* have been read from the video, not how people behave in silence. And muting hides the words but not the act of talking: faces aren't blurred, so a model can still see someone turn and speak.

**Speaker attribution is a prerequisite.** Each microphone hears both people and WhisperX labels are anonymous, so who said a line has to be decided before it can become someone's thought. `speaker_attribution` records how: `llm_inferred` from content and dialogue, `auto_loudness` from the audio, or `manual`.

## 6b. Dense actions and partner influence

Every necessary action by either person is an `action`, including the obvious ones that CoMind never labelled because nobody called them out.

| Field | Meaning |
| --- | --- |
| `relevance` | needed for the current task? `essential`, `optional`, `irrelevant`, or `not_assessed` for imported actions |
| `partner_influence` | `none` (they'd have done it anyway), `influenced`, `unclear`, or `not_assessed`. Use `not_assessed` rather than guessing `none` |
| `influence` | required when influenced: which partner, what triggered it, the cue channels actually used, the related event, and the partner's thought if one explains it |
| `intention_thought_id` | the actor's own thought this carries out, e.g. *I'm going for the knife* → grab knife |
| `predictable_silently` | could someone with no audio have predicted it from what was visible just before? Best judged in a muted pass |
| `predictability_basis` | what made it predictable: task stage, partner action, partner state, partner gaze, object staged, resource state, convention |
| `detection` | `manual`, `comind_import`, or `auto_proposed` (a model suggested it and a person hasn't confirmed it) |

An action is **implicit** when it is influenced and its cue channels are `["none"]`. A `response` or `handover` event's action must be `influenced`; the validator rejects anything else.

**This gives the silent-kitchen test three kinds of action to score separately:**

1. actions that were called out in the real recording: could the cue have been intuited?
2. actions influenced by the partner with no cue: true implicit coordination
3. actions the partner didn't influence: plain task understanding

## 6c. Checkpoints and shared awareness

**Decided 2026-10-20.** The model is given the **recipe** along with the video, for both people. Rationale: with audio on, the objective comes up in conversation early in most real sessions, so assuming it is known is close to what an attentive listener would have. **State this assumption when reporting:** in CoMind only the leader is given the recipe, and the helper works the plan out during the session, so the model starts with what the helper only knows by the end.

**Where the recipe comes from.** Decided 2026-10-22: from the annotations. The recipe the model sees must be the **plan**, not the **performance**. If it is rebuilt from one session's own step labels, it carries that session's actual step order and which steps were skipped, so the model is handed part of what it is asked to infer. Use the recipe as given to the leader if the annotations hold it; otherwise build one canonical recipe per dish, pooled across that dish's sessions, and give every session of the dish the same text.

**Progress is shared, not just done.** A recipe step counts as progress only when it is **done and both people know it is done**. Completion is a fact about the world; awareness is a fact about each person. The gap between them is the coordination signal: short when someone says "onions are done", short without speech when the partner notices, long or failed when nobody notices and the step is duplicated or waited on.

**Checkpoints come from the recipe, per immediate task.** Decided 2026-10-22: once the recipe has been compiled from CoMind's annotations, checkpoints are computed on an immediate-task basis. Each task defines its own checkpoints (what has to be true for that task to be done), so checkpoints are specific to tasks rather than one flat list for the whole dish. The immediate task is the person's innermost task segment (section 3), and each checkpoint links to it through `task_id`. The checkpoint *definitions* come from the compiled recipe, the plan; *when* each one was met comes from the session's labels, and is ground truth, never model input. A checkpoint is **personal** when one person carries it out (`person_ids` has one entry) and **shared** when both do. For a personal step the doer knows it is done when they finish it, so the open question is whether the **partner** noticed.

### Fields, per checkpoint

| Field | Meaning |
| --- | --- |
| `checkpoint_id`, `recipe_step` | the step, as written in the recipe |
| `task_id` | the task-layer segment that carries it out |
| `person_ids` | who carries it out; one entry for personal, two for shared |
| `done_at`, `done_by`, `done_action_id` | when the step finished, who finished it, and the action that finished it |
| `awareness` | one entry per person, below |
| `counted_at` | computed: the latest `aware_by` across both people. This is the progress point. Null while either is unknown |
| `awareness_lag` | computed: `counted_at` minus `done_at` |

### Fields, per person's awareness

| Field | Meaning |
| --- | --- |
| `person_id` | whose awareness |
| `aware_by` | time of the first evidence that they know the step is done. For the doer, `done_at` |
| `evidence` | `said`, `looked`, `acted_on`, `is_doer`, or `none_observed` |
| `evidence_ref` | the thought, CoMind event or action that is the evidence |
| `unaware_evidence` | evidence they did **not** know, each with a time and a ref: `asked` (a question about it) or `redid` (repeated the step) |

### What counts as evidence

| `evidence` | Example | Source in the data |
| --- | --- | --- |
| `said` | "onions are done", "okay" | speech, via the thoughts layer's `acknowledgement` and `belief` thoughts |
| `looked` | the leader glances at the chopped onions | CoMind's joint-attention labels |
| `acted_on` | the leader starts frying, which needs the onions done | the person's next action, read against the recipe order |

### Two rules

1. **Evidence gives "knew by", not "learned at".** If the leader starts frying at 2:10, they knew by 2:10, but may have noticed at 1:50. `aware_by` is an upper bound on when they found out, and every lag computed from it is too.
2. **No evidence means unknown, not unaware.** A person can know and never show it. Only `asked` or `redid` counts as evidence of not knowing. A checkpoint with an unknown awareness has a null `counted_at`; it is never scored as a failure.

### How it is scored

The model gets the **muted video and the recipe**, and at a given time says whether each person knows a given step is done. Answers come from the audio-on labels above. Split the results by evidence type:

1. awareness that came from **speech** (`said`): could it have been inferred without hearing?
2. awareness that came from **looking or acting** (`looked`, `acted_on`): implicit awareness, with no cue at all

## 6d. Sports variant: scored checkpoints (draft, doc only)

**Decided 2026-10-22, not yet in the JSON schema.** For doubles sports, used for objective-level failure grounding: a point is lost about half the time, where kitchen tasks nearly always finish.

**Points are events, checkpoints are windows.** Every point adds positive value (won) or negative value (lost). A checkpoint is a window over a fixed number of total points, and it is never removed or lost: it is evaluated, then the next window starts.

**Soft target.** Inside a window the team should be **in the lead, preferably with 75% of the points**. Graded, not strict pass or fail:

| points won in a window of W | result |
| --- | --- |
| at least 0.75 W | preferred target met, full credit |
| more than half | in the lead, pass with partial credit |
| exactly half | level, narrow fail |
| fewer than half | behind, fail |

A single-number form is the **window margin**, points won minus points lost, from -W to +W.

**Window size scales with the points needed to win.** W is a fixed fraction of the points needed to win the game or set, so a checkpoint means the same across sports and formats. At a fraction of about 0.4:

| format | points to win | W | lead | preferred |
| --- | --- | --- | --- | --- |
| badminton game | 21 | 8 | 5+ | 6+ |
| beach volleyball sets 1 and 2 | 21 | 8 | 5+ | 6+ |
| beach volleyball set 3 | 15 | 6 | 4+ | 5+ |
| table tennis game | 11 | 4 | 3+ | 3+ |

Tennis does not fit points cleanly (a game is 4 points plus deuce), so its unit is **games within a set**, with the same soft target. The fraction is the one free parameter: smaller means more checkpoints and noisier results. Short windows, as in table tennis, are coarse.

**Escalating importance.** Each failed checkpoint makes the next one more important. Open choice: a count of failures, or point importance from win probability (the difference between the chance of winning the match if this point is won and if it is lost), which is the more principled measure. Also open: whether importance is a **training weight** (mistakes in critical windows cost more) or an **input to the agent** (it knows the pressure it is under).

**What a checkpoint does not carry here.** Score awareness is trivial, since both teams know the score, so the shared-awareness fields of 6c are not used. Coordination is judged **within the rally**: does my partner know I am taking this shot, did both or neither go, was a gap left. A lost point is not automatically a coordination failure; separate **coordination failures** (collision, gap, both or neither going) from **individual errors** and **opponent winners**, from player positions where possible.

## 7. Annotation protocol for the pilot

On one session, `cdd2114f` (training split, the richest in CoMind events):

1. **Task pass, audio on, per person.** Goals, then steps inside each goal, then substeps inside each step, the same top-down order Goal-Step uses. Keep a running list of new labels.
2. **Action pass, audio on, per person.** Every necessary action, with `relevance` and `partner_influence`. This is the costly layer: HD-EPIC records roughly 1,400 fine-grained actions per person per hour, against about 36 labelled cued actions per hour in CoMind's training set, so label at the level of object interactions (pick up, put down, open, use) rather than every movement. Time a five-minute clip before committing to a whole session. A model can propose candidate actions first (`detection: auto_proposed`) for a person to confirm.
3. **Thoughts pass.** Speech recast as private thoughts, with speakers attributed. This is the step planned for Claude Opus 5.5; review a sample by hand.
4. **Coordination pass, audio on.** Import the session's CoMind events as starting points and add their task context. Then add what CoMind doesn't have: resource conflicts, and uncued actions that depended on the partner. Where to look:
   - CoMind's implicit handovers and `O`-tagged cued actions
   - actions shortly after the partner finishes a step or moves away from a fixture
   - both people at the same fixture (from the 3D data, once Stage 2 is in)
   - spoken requests for space ("can I get in there")

   **4a. Checkpoint pass, audio on.** For each recipe step: mark `done_at` and `done_by`, then each person's first evidence of awareness, and any `asked` or `redid`. Most evidence is already in the thoughts layer and CoMind's joint-attention labels; the rest comes from reading each person's next actions against the recipe.
5. **Muted pass on a subset.** A second annotator, audio off, fills `visually_resolvable` on events and `predictable_silently` on actions, and marks whether a cue is visible. The disagreement between the muted and unmuted passes is itself a measurement of how much speech carries.
6. **Agreement.** Both annotators independently label about 20% of the session: step-boundary overlap, event-type agreement, and implicit or not. Report it. (Thinking in Space's missing inter-annotator agreement was one of your critiques; this avoids the same gap.)
7. **Validate** every file with the script before it is committed.
8. **Time the pilot.** The hours it takes on one 32-minute session decides whether 55 sessions are feasible, and at what granularity.

## 8. How the labels feed the planned probes

| Probe | Comes from |
| --- | --- |
| "What is my immediate task?" and "What is my partner's?" | the task layer and dense actions, at any time point |
| "What does my partner want?" | the partner's thoughts, which the model is never shown |
| Who is who | `persons`, the verified side mapping, speaker attribution |
| Muted versus unmuted evaluation, cued versus implicit moments | `cue_channels`, `partner_dependency`, `visually_resolvable` |
| "Does my partner know this step is done?" and task progress | the checkpoint layer: `awareness` per person, `counted_at`, `awareness_lag` |
| Collision and future position | computed from the 3D data; `resource_conflict` outcomes give real resolutions to compare predictions against |

## 9. Open decisions for the team

1. **Taxonomy seed.** Start from Goal-Step's 514 cooking steps for compatibility, or grow one from scratch? Goal-Step's release is on [GitHub](https://github.com/facebookresearch/ego4d-goalstep); I haven't checked that its taxonomy file is usable as is.
2. **Substep granularity.** A minimum duration, or only "a change of intent"?
3. **Cue window.** CoMind counts a cue only within 10 s before the action. Keep that?
4. **Leader's actions.** Only at events, as drafted, or densely? The leader is the likelier source of implicit help, because they know the plan.
5. **Recipe text.** Decided 2026-10-22: the full recipe comes from the annotations. Still to confirm: whether the annotations hold the recipe as given to the leader, or the steps as performed in that session. The second leaks the answer (see section 6c, "Where the recipe comes from").
6. **Tool.** CoMind used VIA. If you use it too, a small converter from VIA's export to this format is needed.
7. **Composite sessions.** `real_or_composite` is already in place for conflicts from overlaid sessions; nothing else is designed for composites yet.

## 10. Status of the example file

The worked example uses session `01d63ce5`, 0 to 235 s. **Real:** every time stamp, transcript snippet and `comind_ref` (eight references to four distinct training-set CoMind events, all of which resolve). **Inferred, low confidence:** task labels and boundaries, who spoke, and left = leader, which matches CoMind's statistics (left initiates 70% of training handovers against the paper's 73% for leaders) but hasn't been checked against video. It also shows seven thoughts recast from the real transcript lines (four wishes for the partner, two own intentions, one belief) and the four CoMind-derived actions with their influence fields. It validates cleanly, and twelve deliberately broken variants are each rejected for the right reason.
