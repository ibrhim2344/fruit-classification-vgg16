# Dataset

The supplied notebook names the archive **Fruits-360 original-size** and downloads it from the IBM Skills Network URL in `config.json`. This package does not redistribute the archive or images.

Run `python -m src.download_data`. The downloader detects a complete existing dataset, rejects unsafe ZIP paths, checks ZIP integrity, optionally checks a configured trusted SHA-256, stages extraction, and records the downloaded hash. It does not silently replace an existing incomplete data directory: move it aside or correct it before retrying.

If downloading is unavailable, obtain the archive from an authorized source and extract it under `data/raw/`. Exactly one directory below it must contain `Training`, `Validation` and `Test`. Each split must contain the same nonempty class folders. Both a flat layout and the nested layout of the original notebook are supported. Supported extensions are JPG/JPEG, PNG and BMP.

Class counts and image counts are calculated at runtime in `dataset_summary.json`. No exact count or archive size has been independently verified for this bundle. Dataset inspection checks folder structure and counts; actual image decoding happens in the loading pipeline.

Keep the supplied splits unchanged for a first reproduction. Do not move test images into training or select hyperparameters against test results. Similar views may be correlated; a separate object-level split audit is needed before making strong generalization claims.

See the [upstream Fruits-360 project](https://github.com/fruits-360/fruits-360) and the license bundled with the exact archive you obtain. The repository's MIT license does not grant rights to redistribute dataset images. Preserve the dataset's required attribution. The relationship between the educational mirror and a particular upstream release has not been independently verified.
