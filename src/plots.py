"""Save training curves, a confusion matrix and actual test predictions."""
from pathlib import Path


def plot_history(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head, fine = history["head"], history.get("fine_tune", {})
    boundary = len(head["loss"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        for prefix, label in (("", "Training"), ("val_", "Validation")):
            values = head[prefix + metric] + fine.get(prefix + metric, [])
            ax.plot(range(1, len(values) + 1), values, label=label)
        if fine:
            ax.axvline(boundary + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=metric.title())
        ax.legend(); ax.grid(alpha=0.2)
    figure.tight_layout()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_confusion(matrix, classes, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # Row normalization makes recall failures visible despite unequal class sizes.
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    side = max(8, min(24, len(classes) * 0.35))
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted class", ylabel="True class", title="Test confusion matrix (row normalized)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def plot_samples(generator, predictions, classes, destination, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    indices = np.random.default_rng(seed).choice(len(generator.filepaths), min(6, len(generator.filepaths)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        with Image.open(generator.filepaths[index]) as image:
            ax.imshow(image.convert("RGB"))
        predicted = int(np.argmax(predictions[index]))
        ax.set_title(f"True: {classes[generator.classes[index]]}\nPredicted: {classes[predicted]}\nScore: {predictions[index, predicted]:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)
