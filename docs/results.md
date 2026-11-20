# Results

All runs score the mc_v2 items: three five-option questions per leader instruction (step, action, object), chance 20% each, splits by session. Model runs use Qwen3-VL-4B-Instruct in fp16. Each question is scored from the next-token log-probabilities of the letters A to E, summed over the 5 cyclic rotations of the option order, so letter position cannot help.

## Inputs

- **Blind:** dish name, the numbered recipe steps, minutes elapsed, the question and options. No video, no session history.
- **Zero-shot video:** the same prompt plus two silent clips, the leader's and the helper's head-mounted views, 32 frames each at 1 fps and 320 px, ending 1 s before the leader starts speaking. About 4,000 tokens per prompt.

## Accuracy

| split | question | n | prior | blind | video |
|---|---|---|---|---|---|
| test | step | 135 | 20.0% | 31.9% | 48.9% |
| test | action | 157 | 28.0% | 24.8% | 34.4% |
| test | object | 155 | 18.7% | 34.2% | 39.4% |
| test | all three | 157 | 1.3% | 1.9% | 8.9% |
| dev | step | 89 | | 25.8% | 36.0% |
| dev | action | 118 | | 22.9% | 27.1% |
| dev | object | 115 | | 36.5% | 37.4% |
| dev | all three | 118 | | 1.7% | 5.9% |

## Blind against video, paired on the same items (McNemar exact test)

| split | question | right only with video | right only blind | p |
|---|---|---|---|---|
| test | step | 37 | 14 | 0.002 |
| test | action | 21 | 6 | 0.006 |
| test | object | 20 | 12 | 0.22 |
| test | all three | 11 | 0 | 0.001 |
| dev + test | step | 50 | 18 | 0.0001 |
| dev + test | action | 31 | 11 | 0.003 |
| dev + test | object | 27 | 18 | 0.23 |
| dev + test | all three | 17 | 1 | 0.0001 |

## Reading

- Muted video helps most on which step comes next, and lifts action above the frequency prior, which the blind model could not beat.
- Object barely moves; naming the exact ingredient or tool from small fisheye frames is the hard part.
- Full instructions are still rarely right (under 10%).
- For scale, copying the previous instruction's step (an oracle that uses labels from speech the model never hears) gets 51.1% on test step.

## Caveats

- Test is small, so the 95% margin is about 8 points; the dev gains are smaller than test.
- Labels are model-made and not yet hand-checked.
- One window setting only (32 s at 1 fps, both views). No single-view or window-length ablation yet.
