# Research status and evidence boundaries

This snapshot separates executable model checks from work needed for credible dataset-level experiments. No raw audio, fitted weights, or new benchmark results were added.

## Verified in this update

- The unified CLI help works without importing ML or historical LoRA dependencies.
- Normal and VIB models execute with both configured feature dimensions (32 and 80) on synthetic CPU inputs.
- The self-check exercises finite outputs/loss/gradients, parameter updates, and evaluation.
- CLI failures return nonzero exit codes; checkpoint directories are created for real training.
- Fold summaries use mean and sample standard deviation, and reject missing or non-finite fold results.

## Remaining data and evaluation work

1. **Missing OpenSMILE loader.** `experiment/configs/opensmile_config.py` imports `data.dataset`, but that module is absent. Restore the original schema/loader before claiming a reproducible OpenSMILE dataset workflow.
2. **Dataset-specific MEL labels.** `experiment/audio_mel/dataset.py` reads labels from `data/all_expanded_features_fixed.csv`, then falls back to speaker-number heuristics. Replace that fallback with an explicit, validated label mapping before evaluating a new corpus.
3. **Normalization protocol.** Train and validation datasets each fit their own scaler in the existing loader. For a conventional held-out evaluation, fit on training data and reuse that transformation on validation data. No historical metrics were changed here.
4. **Speaker-fold checks.** Validate that a speaker belongs to exactly one fold, that all required folds exist, and that each evaluation fold has appropriate label coverage. Existing split code is not a substitute for these checks.
5. **No current result claim.** The self-check does not measure classification quality. Report all five real fold measurements and mean ± sample std when data becomes available, with feature extraction, label mapping, and checkpoint-selection policies documented.

## DP scope

`experiment/trainers/core_trainer.py` attaches Opacus to the encoder and updates the classifier with a separate non-private optimizer. It also discards the data loader returned by `make_private_with_epsilon` and trains using the original loader. Sampling/accounting assumptions and the released model's complete optimization path need to be reviewed together.

Accordingly, this repository presents **privacy-oriented experiments**, not a verified end-to-end differential privacy guarantee. VIB information compression is also distinct from differential privacy. Demographic-attribute leakage evaluation is a future direction in the development notes, not an implemented benchmark demonstrated here.

## Historical paths

Whisper LoRA, adversarial code, and the old notebook remain for reference. They are not active features under the repository's development guidance, and their dependencies/quantization/GPU paths were not tested in this update.
