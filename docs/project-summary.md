# Project Summary: Reading a Partner's Intent from What Is Visible

**Updated 2026-10-22 (third revision): CoMind is the core again, with intent prediction as the main objective.** IndEgo and HoloAssist are transfer tests; doubles sports are the extension for objective-level failure. Team: Angshuman Basu and Max (Zilu). Working title. Label details are in [label-schema-implicit-coordination.md](label-schema-implicit-coordination.md).

---

## 1. The idea in one paragraph

Two people work together. A model watches their **muted** first-person video and has to **guess what the other person intends**, from the **world state** and the **visible cues**, mainly gestures. The ground truth is what that person actually said, taken from CoMind's speaker-attributed transcripts. If the model reads intent correctly without hearing, it can then be used to judge and recommend helpful actions; that comes later.

## 2. Core: intent prediction on CoMind

**Data.** CoMind: two people cooking, a leader who knows the recipe and a helper who does not, both on first-person glasses with audio. 55 training sessions.

**Task.** At a moment just before someone speaks, predict the intent their utterance expresses, from muted video up to that moment.

**Ground truth: stated intent.** The thoughts layer recasts every utterance as its speaker's private thought. About **1,800 are intents**: 951 wishes for the partner ("I want the helper to put pepper on the chicken"), 656 own intentions ("I'm going for the knife"), 173 joint intentions ("we should plate it now"). Each is linked to the actions that carried it out and whether it was fulfilled.

**Four rules for a fair test:**

1. **Stop the input before the utterance starts.** Later video leaks the answer, because the partner's reaction shows what was said.
2. **World state, two conditions:** inferred from video (realistic) and given from the annotations (oracle). The gap separates perceiving the state from reasoning about intent.
3. **Scorable intent:** either structured (who, action, object, for whom) or multiple choice against plausible distractors drawn from the recipe and the objects in view.
4. **Results split by cue type.** CoMind labels how each cue was given. Of 1,336 coordination events: 500 speech only, 488 speech and gesture, **210 gesture only**, 89 implicit. Gesture-cued moments stay visible when muted; if they are not clearly easier than implicit ones, the model is not reading visible signals.

**The recipe.** Given as input, compiled from CoMind's annotations as the **plan**, never the session's own performance (that would leak the answer). One canonical recipe per dish if the given recipe is not stored.

## 3. Building blocks for later stages

These are designed and partly labelled; they build on intent prediction:

- **Checkpoints per immediate task,** from the compiled recipe. Progress counts when a checkpoint is done **and both people know it**; knowing is marked by first evidence (said, looked, acted on), and the **awareness lag** is the coordination signal.
- **Help from both sides:** did it advance my step or was it meant for my partner, and did my partner use it.
- **Acting:** first **recommend** actions, scored against an acceptable set (wrong, acceptable, best); later act in a **symbolic simulator built from the labels**, with a partner model learned from CoMind and video aligned to simulator state by cross-attention.

## 4. Transfer tests

| dataset | what transfers | what does not |
| --- | --- | --- |
| **IndEgo** (NeurIPS 2025 D&B) | the **same two-person structure**: partners, teacher and student, or leader and assistant, **both on Aria glasses**, audio and gaze, in a new domain (industrial assembly, repair, logistics, woodworking). Adds **mistake labels** with severity and "help worker" actions. About 197 hours first-person in total; CC BY 4.0 on the dataset card | **no gesture or cue labels**, so the cue-type split runs only on CoMind. Transcripts and speaker labels unclear; collaborative subset size not stated |
| **HoloAssist** (ICCV 2023) | **intent from the world state and timing of help**: a performer in a headset, a remote instructor guiding by voice; 166 hours, 350 pairs; mistakes (about 6% of actions) and **interventions**, with a benchmark predicting intervention from the seconds before the instructor speaks; transcripts, gaze, hand pose; CDLA-Permissive-2.0 | the instructor has **no body in the scene**, so no gestures; one first-person view only; no physical coordination |

## 5. Extension: objective-level failure from doubles sports

Kitchen and workshop tasks nearly always finish, so they teach little about failure of the overall objective. Doubles sports fail often and are cooperative (partner) and adversarial (opponents) at once.

- **Points are positive or negative events; checkpoints are windows** of points, never removed.
- **Soft target:** in the lead within the window, preferably with 75% of its points.
- **Window size scales with the points needed to win,** about 0.4 of them: 8 for a game to 21, 6 for a set to 15, 4 for a table tennis game to 11; tennis uses games within a set.
- **Each failed checkpoint raises the importance of the next,** possibly by point importance from win probability.
- **A lost point is a coordination failure only when it was one:** collision, gap, both or neither going.
- **Data:** BFMD (19 broadcast badminton matches, singles and doubles, 20+ hours, 1,687 rallies, 16,751 hits) and a badminton doubles drone set are third-person; EgoHumans has 2 to 4 first-person wearers (badminton, tennis, volleyball) but about 40 minutes in total and no outcome labels.

## 6. Other datasets considered

| dataset | why not core |
| --- | --- |
| LEMMA | two-person sessions where talking was forbidden, but only 66 short activities; a possible truly silent check |
| CaptainCook4D, EgoPER, Assembly101 | mistakes in single-person procedures |
| Oops! | 20k web failure clips with the moment of failure, mostly third-person and single-person; viewpoint not labelled |
| Harmony4D | close two-person interaction (dancing lead and follow; wrestling, fencing), 3D bodies and contact, no task or speech |
| Ego-Exo4D | single-person drills; pose composites only, for geometry probes |

No dataset labels coordination failures between people on video. That gap is part of the contribution.

## 7. Model sketch

- Frozen video encoder; a small transformer on top, for the semester.
- **Alternating attention** (the idea from VGGT): local windowed attention for the immediate task, global attention over the session for progress. Global attention **causal**.
- Within-person and across-person attention, the analogue of VGGT's within-image and across-image split.
- One token per recipe task or checkpoint, carrying its state.

## 8. Already built

- **Label schema** (doc v0.3.0 plus the 2026-10-22 additions; JSON schema and validator still at v0.2.1).
- **Seed labels for all 55 sessions:** 1,336 coordination events, 89 uncued.
- **Thoughts layer:** 6,432 thoughts across 54 sessions; 42% of helper thoughts depend on hearing the leader, tagged.
- **Session split:** 37 train, 6 dev, 11 test.

## 9. Still needed

1. The intent target: structured fields or multiple choice, and the distractor rule.
2. The pre-utterance input windows, from the transcript timings.
3. The recipe compiled from CoMind's annotations, as a plan.
4. A human review of the thoughts layer: every label so far is model-generated, and speakers are inferred.
5. Check IndEgo's collaborative subset size, transcripts and speaker labels.

## 10. Risks and limits

- **Only stated intent is labelled.** People voice what is not obvious, so the sample is biased toward harder intents; report it as stated intent.
- **Speaking is visible.** A head turn or moving mouth hints that something will be said, not what.
- **Speech can be vague** ("pass me that"); 145 thoughts need the video to resolve the object.
- **Not a claim about silent coordination.** CoMind participants could talk; the claim is reading intent from what is visible.

## 11. Open questions for Lorenzo

1. Is predicting stated intent from muted video a sound core task?
2. Multiple choice or structured intent as the evaluation?
3. Is IndEgo the right transfer test, given it has no cue labels?
4. Is doubles sports worth keeping as the failure extension?
