# Thought labels (schema 0.2.1)

54 of the 55 CoMind training sessions: each is the seed file from [../seed](../seed) plus the thoughts layer (the "silent kitchen"), where everything said aloud is rewritten as what the speaker privately thought, known only to that person.

Missing: `eb784196-2ece-4c95-b430-91a20b985df2`. Writing that session was interrupted partway, and the partial answer was discarded rather than ingested, so it has no file here.

## How it was made

Written by Claude Opus 5.5 in an interactive session, not through the API, by working through the chunk prompts from `scripts/speech_to_thoughts.py build` (822 of the 838 chunks). The answers were checked against the same output schema the API would enforce, then brought in with `ingest` and `merge`. Reads transcripts and CoMind events only, no audio or video.

```bash
python scripts/speech_to_thoughts.py ingest --run runs/thoughts_v1 --dir runs/thoughts_v1/in_session
python scripts/speech_to_thoughts.py merge --run runs/thoughts_v1 --seeds labels/seed --out labels/thoughts --sessions <54 ids, all but eb784196>
python scripts/validate_labels.py labels/thoughts/*.json --comind data/comind/v1/annotations
```

All 54 files pass the validator. The only warnings are the unverified left/right side mapping (caveat 5).

## Heard-dependency tags (helper thoughts)

The helper's thoughts were written from a helper who heard the leader. A model that must not hear the leader should only see the helper thoughts that could exist without hearing. Every helper thought therefore carries `heard_dependency` (added in schema 0.2.1): `independent`, or `depends_on_partner_speech` with a reason.

| Reason | Thoughts |
| --- | --- |
| independent | 1,554 |
| reacts_to_partner_speech | 350 |
| asks_about_partner_speech | 336 |
| answers_partner | 285 |
| acknowledges_partner | 125 |
| uses_info_from_partner_speech | 27 |
| restates_partner_speech | 3 |

That is 1,126 of 2,680 helper thoughts (42%) that depend on the leader's speech. By kind, 118 of 126 acknowledgements and 392 of 1,038 questions depend on it, but only 37 of 252 own intentions do. Across the split, the independent and dependent counts are: train 1,033 / 769, dev 179 / 132, test 342 / 225.

Made by Claude Opus 5.5 in the same interactive way. The tagging read each session as one timeline of both people's thoughts, and judged each helper thought on whether its existence or content needs the leader's words. The answer files live in `runs/heard_dependency_v1/answers` and are applied with:

```bash
python scripts/tag_heard_dependency.py apply --labels labels/thoughts --answers runs/heard_dependency_v1/answers
```

These tags are model-made too, and the line is a judgement. Answers ("yes, it's done") and reactions are counted as dependent because, in the silent kitchen, they reveal when and about what the leader spoke. A study wanting a looser cut can keep `answers_partner` or `reacts_to_partner_speech` thoughts. Leader thoughts are not tagged.

## Counts

| | |
| --- | --- |
| Transcript lines covered | 23,310 of 23,310 (none uncovered, none in two groups) |
| Thoughts | 6,432 (leader 3,752, helper 2,680) |
| By kind | belief 2,878, question 1,543, wish_for_partner 951, own_intention 656, acknowledgement 231, joint_intention 173 |
| Dropped groups | 490 filler, 278 off task, 110 recording talk, 46 unintelligible |
| Groups skipped because the speaker was unclear | 70 |
| Thoughts that need video to resolve a placeholder | 145 |
| Thoughts whose referent comes from a linked CoMind event | 807 |
| CoMind actions linked to a thought that preceded them | 796 of 1,318 |

## Caveats

1. **Not reviewed.** The same model grouped the lines, chose the speakers and wrote the thoughts. Check a sample by hand (start with `runs/thoughts_v1/review_sample.md`) before treating any of it as ground truth.
2. **Speakers are inferred** from content, cross-mic copies and turn-taking (`speaker_attribution: llm_inferred`). WhisperX labels are unreliable:
   - In several sessions one label covers both people on a mic (for example 199d1b52, 1df816f6, 222861ac, 833eac29, 9db1ea2e, b1446007, d90d7d88).
   - In 16b9e645 the dominant label on both mics is the helper.
   - In bb657948 the labels cross so often that most groups are low confidence.

   Confidence is stored per thought.
3. **Links are judgements.** A link means the thought plausibly prompted that CoMind action, based on timing and content. It is not an observed causal fact, and some actions that follow speech were left unlinked when the trigger was unclear.
4. **Questions and acknowledgements presuppose the conversation** (for example "I wonder which ingredient the leader needs"). Together they are 1,774 thoughts, or 28%. Whether a model should see them in the silent setting is an open design decision.
5. **Left = leader is checked for one session only.** Two handovers in cdd2114f (about 48.6 s and 97.6 s) were looked at in the ego video and match the converter's direction. The flag is still unset in the files, and the other sessions are unchecked.
6. **Off-task and setup talk** (chat, comments about the glasses or camera, observers joining in) was dropped, not rewritten. Some sessions also include a third person or an experimenter.

## Attribution

Derived from the CoMind dataset (Gavryushin et al., *CoMind: Understanding Collaborative Human Activity from Multiple Minds and Views*, ECCV 2026, [arXiv:2607.06691](https://arxiv.org/abs/2607.06691)), released under CC BY 4.0. Changes: converted to this schema, transcript speech regrouped across microphones, attributed to a speaker and rewritten as first-person thoughts.
