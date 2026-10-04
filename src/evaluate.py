"""Evaluate every test image; export metrics, predictions and figures."""
import argparse
import csv
from pathlib import Path
from .utils import load_config, read_json, sha256_file, write_json


def evaluate(config, run_dir=None):
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
    from .data import make_generators
    from .plots import plot_confusion, plot_samples

    run = Path(run_dir or config["output_dir"])
    saved = read_json(run / "config.json")
    runtime = {**config, "image_size": saved["image_size"], "batch_size": saved["batch_size"]}
    _, _, test, classes, summary = make_generators(runtime)
    if classes != read_json(run / "class_names.json"):
        raise ValueError("Dataset classes do not match the saved model's class order.")
    model_path = run / "best_model.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    test.reset()
    probabilities = model.predict(test, verbose=1)
    if len(probabilities) != test.samples:
        raise RuntimeError("Prediction count does not match the full test split.")
    truth, predictions = test.classes, probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    metrics = {"status": "evaluated", "test_images": int(test.samples), "num_classes": len(classes),
               "accuracy": float(accuracy_score(truth, predictions)),
               "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)),
               "cross_entropy": float(log_loss(truth, probabilities, labels=labels)),
               "model_sha256": sha256_file(model_path), "image_size": saved["image_size"]}
    report = Path(config["reports_dir"])
    (report / "figures").mkdir(parents=True, exist_ok=True)
    write_json(report / "metrics.json", metrics)
    write_json(report / "dataset_summary.json", summary)
    details = classification_report(truth, predictions, labels=labels, target_names=classes,
                                    output_dict=True, zero_division=0)
    with (report / "classification_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in classes:
            row = details[name]; writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])
    with (report / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["relative_image", "true_class", "predicted_class", "softmax_score"])
        for i, relative in enumerate(test.filenames):
            writer.writerow([relative, classes[truth[i]], classes[predictions[i]], float(probabilities[i, predictions[i]])])
    matrix = confusion_matrix(truth, predictions, labels=labels)
    write_json(report / "confusion_matrix.json", {"classes": classes, "counts": matrix.tolist()})
    plot_confusion(matrix, classes, report / "figures" / "confusion_matrix.png")
    plot_samples(test, probabilities, classes, report / "figures" / "sample_predictions.png", config["seed"])
    print(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config path")
    parser.add_argument("--run-dir", help="Model bundle folder; defaults to config output_dir")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.run_dir)


if __name__ == "__main__":
    main()
