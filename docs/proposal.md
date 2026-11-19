# Reading a Partner's Intent from What Is Visible

**CS 7170 Final Project Proposal**
Angshuman Basu, Max (Zilu)
October 2026

---

## 1. Summary

People who work together constantly read each other without words: one notices the other has finished chopping and starts the pan, or sees a pointed finger and passes the salt. An assistant or robot working beside a person needs the same ability. We propose to measure whether a video model can **predict what a partner intends from muted first-person video**, using the **world state** and **visible cues** such as gestures, before the partner says anything. The ground truth comes for free from speech: in a recorded collaboration with audio, what people said tells us what they wanted. We train and evaluate on CoMind, a dataset of two people cooking together, and test transfer to industrial collaboration (IndEgo) and to instructor-guided tasks (HoloAssist).

## 2. Research questions

1. **Can stated intent be predicted from muted video?** Given muted first-person video up to the moment a person speaks, how often does a model recover the intent their utterance expresses?
2. **Does the model use visible cues?** Are gesture-cued moments, whose cue survives muting, predicted better than moments cued only by speech or not cued at all?
3. **Is the difficulty perception or reasoning?** How much does accuracy improve when the world state (steps done, who holds what) is given rather than inferred from video?
4. **Does it transfer?** Does intent reading learned in kitchens carry over to industrial and instructor-guided collaboration?

## 3. Data

**CoMind [5] (core).** Two people cook together, recorded on first-person glasses with audio: a leader who knows the recipe and a helper who does not. We use its 55 training sessions, split by session into 37 train, 6 dev and 11 test. CoMind labels how each cue was given; of 1,336 coordination events, 500 were cued by speech only, 488 by speech and gesture, 210 by gesture only, and 89 had no cue.

**Intent labels from speech.** We have built a layer that recasts every utterance as its speaker's private thought, linked to the actions that carried it out. About 1,800 of these are intents: 951 wishes for the partner ("I want the helper to put pepper on the chicken"), 656 own intentions, and 173 joint intentions. Per-person task labels follow the goal, step and substep structure of Ego4D Goal-Step [1].

**Transfer sets.**

- **IndEgo** [2]: two workers on industrial tasks (assembly, repair, logistics, woodworking) as partners, teacher and student, or leader and assistant, both wearing Aria glasses; about 197 hours of first-person video with audio, gaze and mistake labels. It has no cue labels, so the cue-type analysis runs on CoMind only.
- **HoloAssist** [3]: a performer in a headset guided by a remote instructor, 166 hours from 350 pairs, with mistake and intervention labels. It tests intent from the world state, not from a visible partner.

## 4. Method

**Task.** At a moment just before a person speaks, the model sees muted video from both participants' views up to that moment, plus the recipe, and predicts the intent the utterance expresses.

**Evaluation format.** Multiple choice: the true intent against plausible distractors built from the recipe and the objects in view. As a secondary measure, structured intent scored field by field: who acts, the action, the object, and for whom.

**Model.** A frozen video encoder with a small transformer trained on top, sized for our compute. The transformer alternates attention over two time scales, local windows for the immediate task and causal global attention over the session for overall progress, and across the two people, within-person and across-person. The alternation follows the frame-wise and global attention design of VGGT [4], applied to time and to people rather than to images.

**Baselines.**

- **Blind:** recipe and answer options with no video, to measure what language priors alone achieve.
- **Prior:** the most frequent intent given the current recipe step.
- **Zero-shot video-language model** on the same multiple-choice questions.
- **Oracle world state:** the true state from the annotations in place of video, as an upper bound on reasoning.

## 5. Evaluation design

- **Stop before the utterance.** The input window ends when speech begins. Later video would leak the answer through the partner's reaction.
- **Recipe as plan, not performance.** The recipe is compiled from CoMind's annotations as the instructions for the dish, never from the session's own steps, which would reveal what happened.
- **Results split** by cue type (gesture, speech, none), by intent kind (wish, own, joint), and by how far ahead of the utterance the prediction is made.
- **Label quality.** All current labels are model-generated. We will hand-check a pilot session, with two annotators on part of it to report agreement, before relying on them.

## 6. Timeline

| dates | work |
| --- | --- |
| Oct 23 to Nov 6 | Hand-check the thought labels on one pilot session; compile recipes; fix the multiple-choice format and distractor rule; build pre-utterance windows |
| Nov 7 to Nov 23 | Extract frozen features; run blind, prior, zero-shot and oracle baselines |
| **Nov 24** | **Mid-semester check-in:** baselines and the cue-type split |
| Nov 25 to Dec 10 | Train the alternating-attention model; IndEgo and HoloAssist transfer |
| **Dec 15 / 17** | Final presentation and paper |

## 7. Risks

- **Only stated intent is labelled.** People voice what is not obvious, so we evaluate on a biased but well-defined slice and will report it as stated intent.
- **Speaking is visible.** A head turn signals that something will be said, though not what; the blind and prior baselines bound how much such signals help.
- **Model-generated labels** may contain errors; the pilot review measures this first.
- **Transfer sets differ in structure:** IndEgo lacks cue labels and HoloAssist has no visible partner, so each tests part of the skill.

## 8. Extensions, if time allows

- **Shared progress:** recipe checkpoints that count only once both people know they are done, with the delay between doing and both knowing as a coordination signal.
- **Recommending actions** that serve the shared objective, scored against the set of acceptable actions.
- **Objective-level failure** from doubles sports, where every point is scored, as a cooperative and adversarial setting.

## References

[1] Y. Song, E. Byrne, T. Nagarajan, H. Wang, M. Martin, L. Torresani. Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities. NeurIPS 2023 Datasets and Benchmarks.
[2] V. Chavan et al. IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants. NeurIPS 2025 Datasets and Benchmarks.
[3] X. Wang et al. HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World. ICCV 2023.
[4] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny. VGGT: Visual Geometry Grounded Transformer. CVPR 2025.
[5] CoMind: arXiv:2607.06691.
