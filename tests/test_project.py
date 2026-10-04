"""Structural/safety tests and an optional offline TensorFlow integration test."""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.download_data import safe_extract
from src.utils import ROOT, inspect_dataset, load_config, read_json, sha256_file, write_json


def fake_tree(root):
    for split in ("Training", "Validation", "Test"):
        for name in ("apple", "pear"):
            folder = root / split / name
            folder.mkdir(parents=True)
            (folder / "example.jpg").write_bytes(b"structural-test-only")


class DatasetTests(unittest.TestCase):
    def test_nested_dataset_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "nested" / "fruits"
            fake_tree(root)
            found, classes, summary = inspect_dataset(temp)
            self.assertEqual(found, root)
            self.assertEqual(classes, ["apple", "pear"])
            self.assertEqual(summary["totals"], {"Training": 2, "Validation": 2, "Test": 2})

    def test_missing_class_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fake_tree(root)
            (root / "Test" / "pear" / "example.jpg").unlink()
            (root / "Test" / "pear").rmdir()
            with self.assertRaisesRegex(ValueError, "do not match"):
                inspect_dataset(root)

    def test_empty_class_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fake_tree(root)
            (root / "Validation" / "pear" / "example.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "No supported images"):
                inspect_dataset(root)

    def test_duplicate_dataset_roots_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fake_tree(Path(temp) / "one"); fake_tree(Path(temp) / "two")
            with self.assertRaisesRegex(ValueError, "found 2"):
                inspect_dataset(temp)

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                safe_extract(archive, Path(temp) / "extracted")
            self.assertFalse((Path(temp) / "escape.txt").exists())

    def test_valid_zip_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "safe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("folder/file.txt", "ok")
            target = Path(temp) / "extracted"
            safe_extract(archive, target)
            self.assertEqual((target / "folder/file.txt").read_text(), "ok")

    def test_json_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.json"
            write_json(path, ["apple", "pear"])
            self.assertEqual(read_json(path), ["apple", "pear"])
            before = sha256_file(path)
            write_json(path, ["pear", "apple"])
            self.assertNotEqual(before, sha256_file(path))

    def test_config_paths_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            write_json(path, read_json(ROOT / "config.json"))
            config = load_config(path)
            self.assertEqual(Path(config["data_dir"]), Path(temp) / "data" / "raw")
            config["image_size"] = [16, 16]
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "dimensions >= 32"):
                load_config(path)

    def test_notebook_code_parses_and_outputs_are_empty(self):
        notebook = read_json(ROOT / "notebooks" / "fruit_classification.ipynb")
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "notebook", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed; integration test skipped")
class TensorFlowSmokeTest(unittest.TestCase):
    def test_training_evaluation_inference_bundle(self):
        # Random-weight training validates execution only, never model quality.
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        from src.data import load_image_batch
        from src.train import train
        from src.evaluate import evaluate
        from src.predict import predict_image
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("Training", "Validation", "Test"):
                for label, color in (("apple", (180, 20, 20)), ("pear", (20, 180, 20))):
                    folder = root / "data" / split / label
                    folder.mkdir(parents=True)
                    for i in range(2):
                        Image.new("RGB", (32, 32), color).save(folder / f"{i}.png")
            config = load_config()
            config.update(data_dir=str(root / "data"), output_dir=str(root / "run"),
                          reports_dir=str(root / "reports"), weights=None,
                          image_size=[32, 32], batch_size=2, head_epochs=1, fine_tune_epochs=1)
            sample = root / "data" / "Test" / "apple" / "0.png"
            actual = load_image_batch(sample, [32, 32])[0, 0, 0]
            np.testing.assert_allclose(actual, [20-103.939, 20-116.779, 180-123.68], atol=1e-4)
            train(config)
            metrics = evaluate(config)
            self.assertEqual(metrics["test_images"], 4)
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertTrue((root / "reports" / "figures" / "sample_predictions.png").is_file())
            prediction = predict_image(sample, root / "run")
            self.assertEqual(len(prediction["predictions"]), 2)
            self.assertEqual({p["class"] for p in prediction["predictions"]}, {"apple", "pear"})
            with self.assertRaises(FileExistsError):
                train(config)


if __name__ == "__main__":
    unittest.main()
