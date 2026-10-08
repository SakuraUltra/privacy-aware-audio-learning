# Privacy-Aware Audio Learning

**Audio classification research with Transformer encoders, differential privacy experiments, and the Variational Information Bottleneck (VIB).**

[简体中文](README.zh-CN.md) · [Run guide](experiment/README.md) · [Examples](experiment/examples.md) · [Research status](docs/research-status.md)

![Audio classification workflow: OpenSMILE or MEL features enter a Transformer, with Normal, encoder-only DP experiments, and VIB paths.](assets/audio-learning-overview.svg)

How do feature representation, information compression, and privacy-oriented training affect audio classification? This PyTorch project explores **OpenSMILE acoustic features**, **MEL spectrograms**, **Transformer classifiers**, and **speaker-aware five-fold cross-validation**.

Previously named `DPIB-MI`. The current research scope is **Normal / DP / VIB**, following the repository's development notes. Historical Whisper LoRA and adversarial code is retained but is not an actively maintained feature.

## Start with a dataset-free check

```bash
git clone https://github.com/SakuraUltra/privacy-aware-audio-learning.git
cd privacy-aware-audio-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python experiment/train_unified.py --help
python experiment/train_unified.py --smoke-test --feature_type opensmile --mode normal
python experiment/train_unified.py --smoke-test --feature_type mel --mode vib
```

The self-check runs a small CPU model on **synthetic feature tensors**: forward pass, loss, backward pass, optimizer update, and evaluation. It downloads no pretrained model and reads no audio or dataset. It checks model execution, not predictive accuracy, data preprocessing, cross-validation, or differential privacy.

`--help` works without ML dependencies. The installation manifest covers the active code paths; it is not an exact historical environment lockfile. See the [verified environment and commands](experiment/README.md#verification).

## What is here?

| Component | Implementation / current status |
|---|---|
| Normal baseline | Transformer encoder, masked pooling, classification head |
| VIB | Gaussian latent representation, KL regularization, beta warmup, Monte Carlo evaluation options |
| DP experiment | Opacus attached to the encoder only; full-model privacy is **not established** |
| Features | 32-dimensional OpenSMILE path; 80-dimensional MEL path in current configurations |
| Evaluation | Speaker-level aggregation and five-fold cross-validation code; summaries report **mean ± sample std** |
| Real-data readiness | Data/checkpoints are not bundled; the OpenSMILE loader is missing from this checkout |

The synthetic model checks cover Normal and VIB with both configured feature dimensions. Full dataset experiments have not been rerun for this update. No benchmark numbers are claimed here.

## Before running real experiments

Read the [run guide](experiment/README.md) and [research status](docs/research-status.md). They describe the expected data layout and known issues in labels, normalization, and the DP training path.

In particular, VIB regularization is not by itself a differential privacy guarantee. The current encoder-only DP implementation also updates a classifier outside Opacus and discards the returned private data loader, so its reported epsilon must not be presented as an end-to-end model guarantee.

## Repository map

```text
experiment/
  train_unified.py       # Single training entrypoint; also provides --smoke-test
  configs/              # Feature and training-mode configurations
  models/               # Transformer and VIB models; historical LoRA code
  trainers/             # Training, speaker-level evaluation, cross-validation
  audio_mel/            # MEL CSV dataset loader
  utils/                # Logging, data handling, self-checks and fold summaries
tests/                  # CLI / model / reporting regression checks
docs/research-status.md  # Evidence boundaries and next engineering steps
requirements.txt        # Core dependency ranges
```

## Useful contributions

- Restore and test the missing OpenSMILE dataset loader with a documented input schema.
- Make labels explicit and fit feature normalization on training folds only.
- Validate speaker-disjoint folds and report all five fold results as mean ± std.
- Rework and audit the DP optimization/data-loader path before claiming privacy guarantees.
- Add reproducible accuracy–privacy-budget comparisons when experiment data is available.

If you work on **audio classification**, **privacy-aware machine learning**, **information bottlenecks**, or **speech representations**, start with the model self-check and the research-status notes.
