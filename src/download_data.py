"""Download once, verify ZIP integrity, and extract into a staged directory."""
import argparse
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from .utils import inspect_dataset, load_config, sha256_file, write_json


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            normalized = member.filename.replace("\\", "/")
            target = (destination / normalized).resolve()
            if not target.is_relative_to(destination) or ":" in normalized or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bad = handle.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP entry: {bad}")
        handle.extractall(destination)


def download_data(config):
    destination = Path(config["data_dir"])
    if destination.exists():
        _, _, summary = inspect_dataset(destination)
        print(f"Dataset already present: {destination}")
        return summary
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination, so an interrupted operation never looks complete.
    with tempfile.TemporaryDirectory(prefix="fruit-download-", dir=destination.parent) as temp:
        temp = Path(temp)
        archive = temp / "dataset.zip"
        request = urllib.request.Request(config["dataset_url"], headers={"User-Agent": "fruit-classification-vgg16/1.0"})
        print("Downloading dataset; the archive can be large...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        checksum = sha256_file(archive)
        expected = config.get("dataset_sha256")
        if expected and checksum.lower() != expected.lower():
            raise ValueError("Dataset SHA-256 does not match config.json.")
        extracted = temp / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        _, _, summary = inspect_dataset(extracted)
        write_json(extracted / "download_manifest.json", {
            "url": config["dataset_url"], "sha256": checksum,
            "expected_checksum_verified": bool(expected), "dataset": summary})
        extracted.rename(destination)
    print(f"Dataset ready: {destination}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config path; defaults to project config.json")
    args = parser.parse_args()
    download_data(load_config(args.config))


if __name__ == "__main__":
    main()
