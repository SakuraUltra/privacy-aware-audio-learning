# Examples

Use the repository's `train_unified.py` entrypoint. Install the root `requirements.txt` first; no shell wrapper is required.

## Check installation without private data

From the repository root:

```bash
python experiment/train_unified.py --help
python experiment/train_unified.py --smoke-test --feature_type opensmile --mode normal
python experiment/train_unified.py --smoke-test --feature_type opensmile --mode vib
python experiment/train_unified.py --smoke-test --feature_type mel --mode normal
python experiment/train_unified.py --smoke-test --feature_type mel --mode vib
```

Expected result: `PASS` for each selected mode, after a CPU forward/backward/optimizer/evaluation check on synthetic features. These runs produce no benchmark results or privacy claims. They do not require the missing OpenSMILE loader.

## Train after preparing real data

See [required files and known limitations](README.md#real-data-prerequisites). Run from `experiment/`, where the default data paths are resolved:

```bash
cd experiment
python train_unified.py --feature_type mel --mode normal --epochs 10 --batch_size 32 --lr 1e-4
python train_unified.py --feature_type mel --mode vib --z_dim 64 --beta 1e-3 --mc_samples 30 --epochs 10
```

OpenSMILE real-data training additionally requires restoring `data/dataset.py` and documenting its input schema. The historical DP command is shown only to identify its interface:

```bash
python train_unified.py --feature_type mel --mode dp --epsilon 8.0 --epochs 10
```

The current DP path is encoder-only and needs a training/data-loader audit; this command is not a validated private-training recipe. LoRA and adversarial examples have been removed from this active guide because the repository marks them unmaintained.

## Regression checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

For scientific results, retain all five fold metrics and report mean ± sample std. Synthetic self-check output is not a substitute for cross-validation.
