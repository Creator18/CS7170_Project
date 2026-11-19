# Seed label files (schema 0.2.0)

One file per CoMind **training** recording (55), generated from CoMind's own human annotations by [scripts/comind_to_labels.py](../../scripts/comind_to_labels.py). They are the starting point for labelling, not finished labels.

## What's in them

| | Count | Source |
| --- | --- | --- |
| Handover events, each with the sender's `pass` action | 350 | CoMind handovers |
| Response events, each with the helper's action | 986 | CoMind cued actions |
| **Implicit events** (derived: no cue, partner-dependent) | **89** | 77 handovers + 12 cued actions |
| Events with no cue type in CoMind (`unknown`, never treated as implicit) | 49 | cued actions |
| Handover and receive pairs linked as the same moment | 129 | matched by object and time |
| Speech in the 10 s before each event, both microphones | 1,306 of 1,336 events | WhisperX transcripts |
| Resources | 572 objects, 94 fixtures | CoMind object categories, one per name per session (not instances) |

Counts match CoMind's training split exactly. Every `comind_ref` resolves to a real, non-hidden CoMind event.

Every imported action is marked `partner_influence: influenced` (that is what CoMind labelled), with its trigger and cue channels; `relevance` and `predictable_silently` are `not_assessed`. The `thoughts` list is empty until the speech-to-thoughts step runs.

## What's missing, and needs someone watching the video

1. **The task layer.** Each file has one placeholder goal, `g0`, so references resolve. Replace it with real goals, steps and substeps, then re-point each action's `task_id` and each event's `task_context`.
2. **Left = leader is unverified.** `side_mapping_verified` is `false` everywhere. Check one session against the side-by-side video; if it's wrong, swap the roles in the converter and regenerate.
3. **Speakers.** Every utterance is `speaker_id: "unknown"`. Both microphones hear both people and WhisperX labels are anonymous.
4. **Resource conflicts, implicit events CoMind didn't label, and the uncalled, obvious actions.** None are generated; dense actions need the video.
5. **Joint attention** isn't converted: it's a shared-attention state, not a coordination event in this schema.
6. 41 handovers carry CoMind's `skip=true`, whose meaning isn't documented. They're kept, marked low confidence.

## Timing

From CoMind's annotator guide: in a cued action, `start_time` is just after the cue completes and `end_time` is just before the action becomes obvious, so the action begins at `end_time`. For handovers the end semantics aren't in the material we read, so treat them as approximate.

## Regenerate and validate

```bash
python scripts/comind_to_labels.py --comind data/comind/v1 --out labels/seed
python scripts/validate_labels.py labels/seed/*.json --comind data/comind/v1/annotations
```

Both need CoMind's annotations and transcripts locally (about 128 MiB; see the download notes).

## Attribution

Derived from the CoMind dataset (Gavryushin et al., *CoMind: Understanding Collaborative Human Activity from Multiple Minds and Views*, ECCV 2026, [arXiv:2607.06691](https://arxiv.org/abs/2607.06691)), released under CC BY 4.0. Changes: converted to this schema, roles inferred from screen side, transcript speech attached by time window.
