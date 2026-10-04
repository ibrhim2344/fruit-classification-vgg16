"""Shared VGG16 preprocessing and split-specific directory iterators."""
from .utils import inspect_dataset


def make_generators(config):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    root, classes, summary = inspect_dataset(config["data_dir"])
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    standard = ImageDataGenerator(preprocessing_function=preprocess_input)
    generators = []
    for split, transform, shuffle in (("Training", train_aug, True),
                                      ("Validation", standard, False),
                                      ("Test", standard, False)):
        generators.append(transform.flow_from_directory(
            root / split, classes=classes, target_size=tuple(config["image_size"]),
            color_mode="rgb", batch_size=config["batch_size"],
            class_mode="categorical", shuffle=shuffle, seed=config["seed"],
            interpolation="bilinear"))
    return (*generators, classes, summary)


def load_image_batch(path, image_size):
    import numpy as np
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.utils import img_to_array, load_img
    image = load_img(path, color_mode="rgb", target_size=tuple(image_size), interpolation="bilinear")
    # Inputs remain in [0, 255] until VGG16 preprocess_input is applied.
    array = img_to_array(image)
    return preprocess_input(np.expand_dims(array, axis=0))
