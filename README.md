# GeoDatum: a small model that calls a verified converter for Nigerian survey coordinates, and an honest look at where it still fails to ask

**One-line claim:** Fine-tuning Tiny Aya Global (3.35B) with LoRA on a free T4 GPU took correct tool calls from 6 of 56 (10.7%) to 56 of 56 (100%) and clean stopping from 24% to 100% on 100 held-out test conversations, but it did **not** learn to ask when the datum or grid is missing (3 of 44 before, 0 of 44 after).

## Tracks and problem statements

Track 1: Fine-tuning and Data Curation.

#6 Reasoning with verifiable answers. Every answer is checked automatically against a converter. Staff confirmed in the Discord ticket (10 Nov 2026) that supervised fine-tuning is accepted for #6. Training here is supervised, not reward-based.

## Abstract

A handheld GPS left on WGS84 when the work was meant to be on the Minna datum put plotted survey points about 100 m from where they belonged. That is the mistake GeoDatum targets. Nigerian survey coordinates are plain numbers: the same easting and northing mean different places depending on the datum and the grid, and a model that guesses the grid gives a confident, wrong answer. GeoDatum fine-tunes Tiny Aya Global (3.35B, 4-bit, LoRA) with supervised learning to call a verified converter for Minna-datum coordinates, to ask when the datum or grid is missing, and to warn when a result falls outside Nigeria. Every answer is checked automatically against a pyproj-based converter, with no judge model. We built a 2,000-row synthetic training set, cleaned it with CuratorKIT, trained with AlignTune on 600 rows for 75 steps (4,664 s on a free Kaggle T4), and evaluated with our own scorer plus AuditKIT on 100 held-out test conversations. Correct tool calls rose from 10.7% to 100% (n=56) and clean stopping from 24% to 100% (n=100). End-to-end correctness rose from 7.1% to 37.5% (n=56). Asking when information is missing did not improve: 6.8% (3 of 44) before and 0% (0 of 44) after, because the model calls the tool on every request and guesses the grid. We report this as the main negative result, with the diagnosis so far (the training loss covered every token, not only the reply) and the next fix to test.

## Headline before → after

100 test rows, same rows, same scorer and same stop rule for both. "Before" is the untrained base model with the same system prompt.

| Metric (higher is better) | Before | After | n |
|---|---|---|---|
| Correct tool call (grid, datum, easting, northing) | 10.7% (6/56) | 100.0% (56/56) | 56 |
| End-to-end correct (right call, right lat/lon, warns exactly when it should) | 7.1% (4/56) | 37.5% (21/56) | 56 |
| Asks (no tool call, no coordinates) when datum/grid is missing or unsupported | 6.8% (3/44) | **0.0% (0/44)** | 44 |
| Clean stop (ends on an end-of-response token) | 24.0% (24/100) | 100.0% (100/100) | 100 |

Base model + method: Tiny Aya Global, LoRA supervised fine-tuning (4-bit) with AlignTune.

With these sample sizes a score near 40% carries roughly ±13 points of uncertainty. 3 of 44 versus 0 of 44 is not a reliable difference, so we write "asking did not improve", not "got worse".

## Problem and approach

Nigerian survey coordinates are plain numbers; the same easting and northing mean different places depending on the datum and grid. Example: E 600000, N 892340 on the Minna datum is

| Grid | Latitude | Longitude |
|---|---|---|
| UTM 31N | 8.07282 | 3.90677 |
| UTM 32N | 8.07283 | 9.90687 |
| West Belt | 12.05042 | 7.88961 |
| Mid Belt | 12.07030 | 7.85110 |
| East Belt | 12.03168 | 7.81597 |

A model that guesses the grid gives a confident, wrong answer. GeoDatum is trained to (1) ask when the datum or grid is missing, (2) call a verified converter instead of computing, and (3) warn when a result falls outside Nigeria. Minna has no UTM zone 33N in the standard EPSG registry, so a request for it must be refused and redirected. **Result: (2) was learned; (1) was not; (3) was learned only partly (see End-to-end above).**

## Why this problem / real use case

While working for the Surveyor General of Kwara State, Nigeria, my boss sent me to pick coordinates with a handheld GPS. I forgot to change the GPS to the Minna datum, so the readings were in WGS84. When I plotted the points in AutoCAD and checked them against Google Earth, they did not sit where they should: they were about 100 m away from the real location. We then found that the datum was the cause.

An application built on this model should warn the user when the datum or grid is missing, or when a point falls outside Nigeria, and tell them what to do. The same mistake can happen in any country that uses a local datum, and in any work that moves coordinates between datums. This is the behaviour we targeted; the current model does the conversion part but does not yet reliably ask (see Results). These are intended users; no real users have tested it.

## Hackathon repos (kits) used

- **CuratorKIT:** schema check, exact deduplication and cleaning, then export to chat-message format. `clean_transforms={"strip_html": False}` because the default strips the `<tool_call>` tags. Result: 2,000 passed, 0 rejected; rows containing `<tool_call>`: 1,498 in and 1,498 out; 0 rows with changed text. By first assistant turn, 1,042 rows start with a tool call and 958 start with an ask. CuratorKIT removed nothing because the generator already produces unique, clean rows.
- **AlignTune:** `create_sft_trainer` with the TRL backend, LoRA (`peft_enabled=True`) and 4-bit loading, system prompt passed through its `system_prompt` argument.
- **AuditKIT:** `ak.evaluate` with `exact_match` scored the saved generations on four metrics, and `ak.compare` compared before and after. Generation and parsing are our own code (`score.py`); scoring and comparison are AuditKIT's. Every AuditKIT result was cross-checked against `score.py`'s own flags and agreed on every row. We did not use AuditKIT's agent-evaluation layer.

How the repos connect: generator (pyproj converter as answer key) → CuratorKIT (check, dedupe, export) → AlignTune (LoRA SFT) → `score.py` generates replies → AuditKIT scores and compares.

## Models and licenses

- Base: `CohereLabs/tiny-aya-global`, 3,356,895,232 parameters, loaded in 4-bit. LoRA trainable parameters: 7,667,712 (0.2284%). Licence: CC-BY-NC with Cohere's acceptable use policy (see the model card). Non-commercial.
- Kits: source-available, non-commercial licences (see each repo).
- Closed-model baseline: none.

## Datasets and licenses

- Source: synthetic, generated by `generate_dataset.py` from the converter. No client data, no real survey plans. Licence: CC-BY-4.0.
- Language: English only.
- Splits: 2,000 train / 600 test. The test set excludes any first user message that appears in train (exact-match leak check only; no embedding or MinHash check). The data is templated, so test rows share phrasing patterns with train. Only the first 100 test rows were scored.

| Scenario | Train | Test |
|---|---|---|
| complete_inside | 749 | 219 |
| complete_outside | 293 | 71 |
| missing_grid | 287 | 97 |
| missing_datum | 299 | 86 |
| missing_both | 182 | 64 |
| unsupported_zone (33N) | 190 | 63 |

Outside-Nigeria rows, by cause (train / test): random noise 101 / 32, wrong grid label 70 / 13, swapped easting and northing 62 / 14, dropped digit 60 / 12.

## Metrics, and why these

All metrics are checked automatically against the converter; no judge model. Higher is better. (1) Tool-call arguments, because a wrong grid is the core error. (2) End-to-end, because the user needs the right coordinates and the right warning. (3) Asking, because the model must not guess. (4) Clean stop, because a model that does not end its turn cannot be used in a tool loop.

## Method and hyperparameters

- Hardware: free Colab T4 (timing run) and free Kaggle T4. Compute cost: $0. Training took 4,664 s (about 1.3 GPU-hours) for 600 rows; scoring 100 test rows took 893 s for the trained model and 1,582 s for the base model.
- AlignTune: `model_name="CohereLabs/tiny-aya-global"`, `backend="trl"`, `num_epochs=1`, `batch_size=2`, `gradient_accumulation_steps=4`, `max_samples=600`, `max_seq_length=768`, `peft_enabled=True`, `quantization={"load_in_4bit": True}`, `learning_rate=2e-4` (read from the saved `training_args.bin`), `config_name="sft_messages"`. That is 75 optimiser steps.
- Training loss (logged every 10 steps): 2.73, 1.60, 0.79, 0.24, 0.17, 0.16, 0.15. Loss alone proved misleading: the data is templated, so it fell fast while the ask behaviour was not learned.
- Loss masking: the saved training arguments show `assistant_only_loss=False` and `completion_only_loss=None`, so the loss covered every token (system prompt, user message and reply), not only the assistant's reply.
- Only the first 600 of the 2,000 curated rows were used, because of free-GPU time limits. The first 600 rows are balanced (308 tool-call rows, 292 ask rows).
- One run, one seed. No ablations.
- Evaluation: the first 100 test rows, same rows for before and after, greedy decoding, up to 200 new tokens. Generation stops on token ids [3, 6, 261001] (`<EOS_TOKEN>`, `<|END_OF_TURN_TOKEN|>`, `<|END_RESPONSE|>`).

## Results

Headline table above. By scenario (after the adapter, 100 test rows):

| Scenario | What is scored | Before | After | n |
|---|---|---|---|---|
| complete_inside | end-to-end correct | 7.5% | 40.0% (16/40) | 40 |
| complete_outside | end-to-end correct | 6.2% | 31.2% (5/16) | 16 |
| missing_grid | asks | 0.0% | 0.0% | 10 |
| missing_datum | asks | 7.7% | 0.0% | 13 |
| missing_both | asks | 16.7% | 0.0% | 12 |
| unsupported_zone | asks | 0.0% | 0.0% | 9 |

Outside-Nigeria rows by mistake type, end-to-end correct after training: noise 3 of 9, swapped easting/northing 1 of 3, wrong grid 0 of 3, dropped digit 1 of 1 (5 of 16 in total; 1 of 16 before). These groups are very small.

`ak.compare` per-row changes (before → after): tool calls 50 rows wrong→correct, 0 correct→wrong; clean stop 76 and 0; end-to-end 19 and 2; asking 0 and 3.

## Sample inputs and outputs

Replies from `after.json` (the trained adapter) on test rows. The scenario of each row is shown.

| Scenario | User message | Model reply | Verdict |
|---|---|---|---|
| complete_inside | I have a beacon at E: 1165283, N: 1082239 (Nigeria East Belt, Minna datum). Where is it in lat/lon? | Tool call with `East Belt`, easting 1165283, northing 1082239. Final answer: "The beacon is located at latitude 13.7875 and longitude 13.0073, within Nigeria's East Belt." | correct |
| complete_outside | I have a beacon at E 515468.96 N 90229.40 (UTM zone 31 north, Minna datum). Where is it in lat/lon? | Tool call with `UTM 31N`. Final answer: "This beacon is outside Nigeria. It is in UTM zone 31N, Minna datum, but at latitude 0.81736 and longitude 3.13832, which is outside Nigeria's border." | correct |
| missing_grid | Please convert X = 732829, Y = 1146479 (Minna) to lat/lon. | Tool call with grid `UTM 31N`. The grid was never given. | **failed: should ask** |
| unsupported_zone | Convert X = 537857.99, Y = 854350.39, Minna datum, UTM zone 33N, to latitude and longitude. | Tool call with grid `UTM 33N`, copied from the user. The tool does not have this grid. | **failed: should refuse and ask** |
| complete_inside | (see `test.jsonl`) | Correct tool call. Final answer: "E 693195 N 1437747 is in Nigeria." | **failed: no latitude/longitude** |
| complete_outside | (see `test.jsonl`) | Correct tool call. Final answer: "This point is outside Nigeria. Please check the grid and datum." | **failed: warning right, no coordinates** |

In the end-to-end failures we inspected, the warning logic was right but the latitude and longitude were missing from the answer. This is a small sample, so we treat it as a pattern to check, not a finding. The same guessing happened on prompts taken from the training set (3 of 3 missing-grid and 3 of 3 missing-datum prompts got a tool call).

## What didn't work (negative results)

- **The model never learned to ask.** After training it starts every reply with a tool call, guesses `UTM 31N` for a missing grid, fills in `Minna` for a missing datum, and copies an unsupported `UTM 33N` from the user. The training data is balanced (958 of 2,000 rows are ask rows) and the cleaned file keeps them all, so missing data is not the cause. Two hypotheses remain **untested**: (a) our run had `assistant_only_loss=False`, so the loss covered the system prompt and the random coordinate digits in the user turns as well as the replies; whether that explains the failure is untested; (b) 75 steps on 600 rows may simply be too little training.
- A first 200-row adapter never asked and repeated the same tool call. Cause: the model ended its response with `<|END_RESPONSE|>` and then started a new turn, and our scorer did not stop on that token. We added it to the stop list.
- The first system prompt listed grids as "Nigeria Mid Belt" while the tool expects "Mid Belt", so the model copied the wrong name. We changed the prompt to list the exact names.
- The untrained model's first scores used the old stop rule, so they were re-run with the fixed scorer. The numbers in this README are the re-run, final ones.
- Our first AuditKIT script showed the "after" column equal to the "before" column. AuditKIT cached the first run's result and our two runs had identical fingerprints. We noticed because the numbers disagreed with `score.py`, fixed it by disabling the cache and naming each run, and added an automatic cross-check that stops the script if AuditKIT and a direct count differ.

## Limitations

- Synthetic, templated data: the test set is in-distribution, so scores overstate real-world performance.
- All training data uses the Minna datum.
- "Inside Nigeria" is a coarse bounding box, not a border polygon.
- Only 600 of 2,000 rows trained, 100 test rows scored (56 and 44 per metric), one run.
- The trained model guesses a missing grid or datum, so it should not be used where those might be missing.
- No real users have tested it. The intended users listed above are hypotheses.

## Suggestions for improvement

These are ordered by how directly they address the main failure (the model does not ask). They are things to test, not results. Each one names the check that would show whether it worked: `ask_correct` on the same 100 rows, with the other three metrics watched for regressions.

1. **Compute the loss on the assistant's reply only.** Our run had `assistant_only_loss=False`, so the loss covered every token, including text that is identical in every row (the system prompt) and random digits that cannot be predicted. That may have diluted the signal for the ask-or-call decision. Turning the option on is the cheapest fix to try, but it needs a chat template that marks the assistant's tokens, which we have not checked for this model; the alternative is to convert the data to a prompt-completion format.
2. **Train longer, on all 2,000 rows.** The current run used 600 rows for 75 steps. More steps and more of the ask examples may be enough on their own. Running 1 and 2 separately would show which one mattered.
3. **Watch the ask behaviour during training.** Training loss fell to 0.15 while `ask_correct` stayed at 0%, so loss alone was misleading. Scoring a small ask-only set every N steps would show early whether the model is learning to ask, and would allow stopping a bad run.
4. **Add contrast pairs.** Show the same coordinates twice: once complete (tool call) and once with the grid or datum removed (ask). That makes the missing piece the only difference the model can use.
5. **Reward-based training (GRPO) with the converter as the reward.** Reward asking when information is missing and calling when it is complete, and penalise a guessed grid. This fits problem statement #6 directly.
6. **A check outside the model.** A simple rule that looks for a grid and datum in the user's message before the model is called would catch the mistake even while the model is weak, and a tool-side check could refuse any call whose arguments the user never gave.
7. **Cover the mistake that started this project.** Add WGS84 as an input datum and teach the model to flag a datum mismatch, for example a GPS reading in WGS84 that is being treated as Minna.
8. **Messier, more realistic inputs.** Real phrasing, units, degree-minute-second formats and typos, since the current test rows are templated and in-distribution.
9. **A stronger evaluation.** Score all 600 test rows instead of 100, and repeat with more than one seed, so the numbers carry tighter intervals.
10. **Smaller things.** Replace the bounding box with a border polygon, add other local datums and countries, and re-test the "outside Nigeria" warning, which only partly works today (end-to-end 37.5%).

## Artifacts

- HF model: https://huggingface.co/commandofss/geodatum-curated (collection: https://huggingface.co/collections/commandofss/geodatum-models)
- HF dataset: https://huggingface.co/datasets/commandofss/geodatum-data (collection: https://huggingface.co/collections/commandofss/geodatum-dataset)
- HF dashboard: https://huggingface.co/spaces/commandofss/geodatum-dashboard
- GitHub: https://github.com/commandofss/geodatum
- Demo / video: https://youtu.be/FyYnG5suGIQ
## Reproduction

```bash
pip install pyproj curatorkit aligntune peft bitsandbytes auditkit
python geodatum_converter.py            # self-test, expect ALL OK
python generate_dataset.py --n 2000 --seed 1 --out train.jsonl
python generate_dataset.py --n 600 --seed 2 --out test.jsonl --exclude train.jsonl
python curate.py train.jsonl output/clean_messages   # expect CURATION CHECK: PASS
# train with the create_sft_trainer call in the notebook, then:
python score.py --test test.jsonl --label after  --adapter ./run1 --limit 100 --out after.json
python score.py --test test.jsonl --label before --limit 100 --out before.json
python audit_eval.py --test test.jsonl --before before.json --after after.json --out audit_results.json
```

Files: `geodatum_converter.py` (answer key and tool), `generate_dataset.py`, `curate.py`, `score.py`, `audit_eval.py`, notebooks. Results: `results/before.json`, `results/after.json`, `results/audit_results.json` in the dataset repo.

## Declaration of work done outside the hackathon

All code and data in this repo were written between 9 and 13 November 2026. No earlier code is reused; the converter was written fresh for this project on 10 November.

## Teammates

None (solo).


