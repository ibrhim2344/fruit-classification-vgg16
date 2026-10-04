# Validation performed for this source release

Date: 2026-10-04

The pinned runtime in `requirements.txt` was installed into an isolated Linux Python 3.12 environment. `python -m pip check` reported **No broken requirements found**. Exact installed packages are listed in `validated-environment.txt` (an environment record, not a cross-platform lockfile).

## Results

- **10 tests passed, 0 skipped** using `python -m unittest discover -s tests -v`.
- Dataset nesting/counts, class mismatch, empty classes and duplicate dataset roots were checked.
- Safe ZIP extraction and rejection of parent-directory traversal were checked.
- Configuration path resolution, JSON persistence and content hashing were checked.
- Notebook code cells were parsed and verified to contain no fabricated outputs.
- All four command-line help entry points worked without importing TensorFlow eagerly.
- Python source syntax and documentation control characters were checked.
- All three figure-generation functions rendered valid PNG files using temporary synthetic data.
- The TensorFlow integration test completed both training stages on a tiny synthetic image dataset, saved/reloaded the model, evaluated all four synthetic test images, exported reports and performed inference.
- VGG16 preprocessing was numerically checked against the expected RGB-to-BGR channel order and mean subtraction.
- Training refused to overwrite a completed run directory.

The smoke test used random VGG16 weights (`weights=null`), two classes, 32 × 32 images and one epoch per stage. It establishes basic pipeline execution, not model quality. Its temporary weights, figures and metrics are deliberately not distributed as fruit-classification results.

The local test set TensorFlow/OpenMP thread counts to 1 and disabled GPU visibility. The test class itself limits TensorFlow inter/intra-op threads. The runtime emitted upstream directory-generator and plotting deprecation warnings; no test failed. These warnings were not globally suppressed.

## Not performed

- Downloading and independently auditing the complete Fruits-360 mirror/archive.
- Training with pretrained ImageNet weights on the actual fruit dataset.
- Measuring full-dataset accuracy, checking object-level data leakage or validating field-photo performance.
- Running the full interactive notebook against real data.
- Testing Windows/macOS or GPU training.
- Executing GitHub Actions remotely; the workflow will run after the repository is published.

The notebook frontend requirements are provided separately; the tested installation was the runtime requirements. Complete the real-data workflow before filling the README results table or publishing trained weights.
