# Fruit Classification with VGG16

A reproducible computer vision project for classifying produce images with **transfer learning** and **fine-tuning**. Built from an exploratory TensorFlow notebook, with separate commands for downloading data, training, full-test evaluation and single-image inference.

**Author:** Ibrahim Al-Hashimi · [الدليل العربي](README_AR.md)

> **Project status:** source package prepared; no trained model, benchmark accuracy or real prediction images are bundled. Run training and evaluation to produce these artifacts. This is an educational portfolio project, not a validated production system.

## What the project does

- Loads the Fruits-360 original-size archive linked in the supplied notebook.
- Verifies the same class folders exist in Training, Validation and Test.
- Resizes RGB images to 64 × 64 and applies VGG16's ImageNet preprocessing.
- Trains a classification head while freezing VGG16, then fine-tunes `block5`.
- Selects the best model across both stages using full-validation loss.
- Exports full-test accuracy, macro/weighted F1, per-class metrics and predictions.
- Saves training curves, a confusion matrix and six test-image predictions.

## Model

| Component | Setting |
| --- | --- |
| Backbone | VGG16, ImageNet weights, original classifier removed |
| Input | 64 × 64 RGB, bilinear resizing |
| Preprocessing | `vgg16.preprocess_input` on pixel values in [0, 255]; no `/255` |
| Head | GlobalAveragePooling → Dense(256, ReLU) → BatchNormalization → Dropout(0.3) → Softmax |
| Stage 1 | Frozen backbone, Adam learning rate 0.001, up to 5 epochs |
| Stage 2 | Trainable block5, Adam learning rate 0.00001, up to 5 epochs |
| Model selection | Lowest validation cross-entropy across both stages |

64 × 64 preserves the original experiment's input size and reduces computation. It may discard fine detail. Comparing it with larger resolutions is a future experiment, not an established improvement. VGG16 remains relatively heavy; a GPU is useful for training, while CPU execution can be slow.

## Setup

Use **Python 3.12** and run commands from the repository root. These dependency pins are the target environment, not a claim that all platforms have been validated. See [validation status](docs/VALIDATION.md).

```bash
python -m venv .venv
```

Activate on Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Notebook users can instead install `requirements-notebook.txt`, which includes the runtime requirements.

## Run the complete workflow

```bash
python -m src.download_data
python -m src.train
python -m src.evaluate
```

Downloading needs internet access and sufficient disk space. The first ImageNet model initialization also downloads weights. The dataset link is inherited from the source notebook and may become unavailable; see [data instructions](data/README.md) for manual placement.

Each command accepts `--config path/to/config.json`. Relative paths in a configuration file resolve from the directory containing that file. Relative command-line paths such as `--image` and `--run-dir` resolve from your current working directory.

Training refuses to overwrite a nonempty run directory. For a new experiment, change `output_dir` and preferably `reports_dir` in your config. The reported best model can come from stage 1 if fine-tuning does not improve validation loss.

## Predict a new image

After training, replace `path/to/fruit.jpg` with an actual image on your computer:

```bash
python -m src.predict --image path/to/fruit.jpg --top-k 3
```

To use a different model bundle or save predictions:

```bash
python -m src.predict --image path/to/fruit.jpg --run-dir models/run_vgg16 --output prediction.json
```

The output lists class names and softmax scores. A high score is not proof of correctness. The model always chooses among known training classes and has no unknown-class rejection mechanism.

## Notebook

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook notebooks/fruit_classification.ipynb
```

The notebook imports the same modules as the command-line workflow. It includes dataset counts, class balance, training, evaluation and visualization without maintaining a second copy of the pipeline. Its cells are intentionally unexecuted in this package.

For Colab, upload and extract this project ZIP, change the working directory to its root, install the requirements, and execute the notebook. If Colab reports that already-imported packages changed, restart the runtime before continuing. Its preinstalled Python/package versions may require a separate compatible environment; Python 3.12 locally is the declared target.

## Files and generated artifacts

| Path | Purpose |
| --- | --- |
| `config.json` | Experiment settings and paths |
| `src/utils.py` | Configuration, dataset checks and JSON helpers |
| `src/download_data.py` | Staged download, ZIP verification and safe extraction |
| `src/data.py` | Shared image preprocessing and split iterators |
| `src/model.py` | Architecture and fine-tuning policy |
| `src/train.py` | Two-stage training and validation-based checkpoint selection |
| `src/evaluate.py` | Full-test evaluation and exported predictions |
| `src/predict.py` | Single-image inference |
| `src/plots.py` | Training, confusion and sample-prediction figures |
| `notebooks/fruit_classification.ipynb` | Explained interactive workflow |
| `data/raw/` | Downloaded data; generated and ignored by Git |
| `models/run_vgg16/` | Model bundle; generated and ignored by Git |
| `reports/` | Metrics and figures; generated after actual runs |
| `tests/` | Dataset/extraction tests and optional TensorFlow smoke test |
| `.github/workflows/tests.yml` | Automated tests for pushes and pull requests |

The exported model bundle contains `best_model.keras`, `class_names.json`, `config.json`, `history.json`, `selection.json`, `dataset_summary.json`, `environment.json`, and stage checkpoints/CSV histories. Keep the model, class names and configuration together for inference.

## Results

**Not yet measured for this refactored pipeline.** Preprocessing differs from the original notebook, so previously obtained numbers must not be presented as results for this version.

After evaluation, use the actual values in `reports/metrics.json` to complete this table:

| Metric | Result |
| --- | --- |
| Test images | Not measured |
| Test accuracy | Not measured |
| Macro F1 | Not measured |
| Weighted F1 | Not measured |
| Test cross-entropy | Not measured |

Generated figures are written to `reports/figures/training_curves.png`, `confusion_matrix.png` and `sample_predictions.png`. Embed them here after generating and reviewing them. No synthetic examples are presented as model results.

## Validation and reproducibility

```bash
python -m unittest discover -s tests -v
```

The TensorFlow smoke test uses a tiny synthetic dataset and `weights=null` to check plumbing without downloading pretrained weights. It is not an accuracy benchmark. If TensorFlow is not installed, this test is explicitly skipped. CI installs the pinned runtime and runs the entire suite; its outcome is known only after GitHub runs it.

A fixed seed and deterministic TensorFlow operations reduce variability; hardware, library versions and input data can still affect reproducibility. Set `dataset_sha256` in your config to the trusted archive hash if you have one. By default the downloader records a hash but does not assert an independently verified dataset identity. Model hashes are recorded in evaluation results.

## Limitations

- Performance on this dataset does not establish reliability on field, shop or phone photos.
- Multiple views of similar fruit can create correlations; this project retains the supplied splits and does not claim to have audited object-level leakage.
- Dataset-wide counts are computed at runtime; no class count is hard-coded.
- Softmax scores are uncalibrated and cannot detect unknown fruit or non-fruit inputs.
- The supplied dataset includes produce categories such as cucumber; this is not a botanical taxonomy system.
- The directory-generator API is retained for continuity with the original notebook and pinned environment. A future migration to `tf.data` should be separately validated.

## Attribution and license

The starting point was the user-supplied **Fruit Classification Using Transfer Learning .ipynb**. Its dataset URL points to IBM Skills Network storage. No claim of affiliation, dataset ownership, or authorship of third-party course material is made. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

New project code and documentation are provided under the [MIT License](LICENSE). Dataset images, pretrained weights and any separately sourced educational materials retain their own licenses.

References:

- [Keras VGG16 documentation](https://keras.io/api/applications/vgg/)
- [Keras transfer learning guide](https://keras.io/guides/transfer_learning/)
- [Fruits-360 upstream project](https://github.com/fruits-360/fruits-360)

## Publishing to GitHub

Follow [docs/GITHUB_GUIDE_AR.md](docs/GITHUB_GUIDE_AR.md). Publish the extracted source folder rather than uploading only this ZIP. Add verified results after running the pipeline. Upload model bundles separately if you choose to distribute them; do not commit the full dataset.
