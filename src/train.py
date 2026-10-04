"""Train a frozen head, fine-tune a VGG block, and export a reproducible bundle."""
import argparse
from pathlib import Path
from .utils import load_config, write_json


def train(config):
    import platform
    import numpy as np
    import tensorflow as tf
    from .data import make_generators
    from .model import build_model, enable_fine_tuning
    from .plots import plot_history

    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}. Set a new output_dir in config.json.")
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    train_gen, val_gen, _, classes, summary = make_generators(config)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "class_names.json", classes)
    write_json(output / "dataset_summary.json", summary)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": tf.keras.__version__, "numpy": np.__version__,
        "gpus": [device.name for device in tf.config.list_physical_devices("GPU")]})
    model, base = build_model(config, len(classes))
    model.summary(print_fn=lambda line, **kwargs: print(line))
    histories, candidates = {}, []
    for phase, epochs, rate in (
        ("head", config["head_epochs"], config["head_learning_rate"]),
        ("fine_tune", config["fine_tune_epochs"], config["fine_tune_learning_rate"])):
        if phase == "fine_tune":
            # Start fine-tuning from the best validation checkpoint of the first stage.
            model.load_weights(output / "head.keras")
            enable_fine_tuning(base, config["fine_tune_block"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rate),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        checkpoint = output / f"{phase}.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"], restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                patience=config["reduce_lr_patience"], min_lr=1e-7),
            tf.keras.callbacks.CSVLogger(str(output / f"{phase}_history.csv")),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        # No hard-coded step limits: each epoch covers the full split.
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
        histories[phase] = {key: [float(v) for v in values] for key, values in history.history.items()}
        values = histories[phase]["val_loss"]
        if not checkpoint.exists() or not all(np.isfinite(values)):
            raise RuntimeError("Non-finite validation loss or missing checkpoint; inspect input data and logs.")
        candidates.append((min(values), phase, checkpoint))
        write_json(output / "history.json", histories)
    best_loss, phase, checkpoint = min(candidates, key=lambda item: item[0])
    best = tf.keras.models.load_model(checkpoint)
    best.save(output / "best_model.keras")
    write_json(output / "selection.json", {"criterion": "minimum full-validation loss across both stages",
               "selected_stage": phase, "validation_loss": best_loss})
    plot_history(histories, Path(config["reports_dir"]) / "figures" / "training_curves.png")
    print(f"Saved {output / 'best_model.keras'} (selected stage: {phase})")
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config path")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
