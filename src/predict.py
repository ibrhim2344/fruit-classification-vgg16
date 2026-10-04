"""Predict one image using the preprocessing and class order saved with a model."""
import argparse
import json
from pathlib import Path
from .utils import load_config, read_json, write_json


def predict_image(image_path, run_dir, top_k=3):
    import numpy as np
    import tensorflow as tf
    from .data import load_image_batch
    image_path, run = Path(image_path), Path(run_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    config = read_json(run / "config.json")
    classes = read_json(run / "class_names.json")
    model = tf.keras.models.load_model(run / "best_model.keras", compile=False)
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output size and class_names.json do not match.")
    scores = model.predict(load_image_batch(image_path, config["image_size"]), verbose=0)[0]
    indices = np.argsort(scores)[::-1][:min(top_k, len(classes))]
    return {"image": image_path.name, "predictions": [
        {"class": classes[int(i)], "softmax_score": float(scores[i])} for i in indices],
        "note": "Scores are not calibrated certainty; unknown classes are not detected."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--config", help="JSON config path")
    parser.add_argument("--run-dir", help="Model bundle folder")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    config = load_config(args.config)
    result = predict_image(args.image, args.run_dir or config["output_dir"], args.top_k)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
