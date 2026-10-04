"""Configuration, dataset discovery and portable metadata helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path=None):
    path = Path(path).resolve() if path else ROOT / "config.json"
    config = read_json(path)
    for key in ("data_dir", "output_dir", "reports_dir"):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if len(config["image_size"]) != 2 or min(config["image_size"]) < 32:
        raise ValueError("VGG16 requires image_size [height, width] with both dimensions >= 32.")
    for key in ("batch_size", "head_epochs", "fine_tune_epochs"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["weights"] not in ("imagenet", None):
        raise ValueError("weights must be 'imagenet' or null (null is for offline smoke tests).")
    return config


def find_dataset_root(directory):
    """Support both flat and nested versions of the downloaded archive."""
    directory = Path(directory)
    candidates = [directory] + sorted({p.parent for p in directory.rglob("Training") if p.is_dir()})
    matches = [p for p in candidates if all((p / split).is_dir() for split in ("Training", "Validation", "Test"))]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(f"Expected one dataset root with Training/Validation/Test under {directory}; found {len(matches)}.")
    return matches[0]


def inspect_dataset(directory):
    """Check class alignment and count supported images in all three splits."""
    base = find_dataset_root(directory)
    classes = sorted(p.name for p in (base / "Training").iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("At least two training classes are required.")
    counts = {}
    for split in ("Training", "Validation", "Test"):
        observed = sorted(p.name for p in (base / split).iterdir() if p.is_dir() and not p.name.startswith("."))
        if observed != classes:
            raise ValueError(f"Class folders in {split} do not match Training.")
        counts[split] = {}
        for name in classes:
            count = sum(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in (base / split / name).rglob("*"))
            if count == 0:
                raise ValueError(f"No supported images in {split}/{name}.")
            counts[split][name] = count
    return base, classes, {"class_names": classes, "num_classes": len(classes),
                           "counts_per_class": counts,
                           "totals": {split: sum(items.values()) for split, items in counts.items()}}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
