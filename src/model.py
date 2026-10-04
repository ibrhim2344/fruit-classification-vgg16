"""VGG16 backbone and a small classification head."""


def build_model(config, num_classes):
    from tensorflow import keras
    base = keras.applications.VGG16(
        weights=config["weights"], include_top=False,
        input_shape=(*config["image_size"], 3))
    base.trainable = False
    inputs = keras.Input(shape=(*config["image_size"], 3), name="preprocessed_image")
    x = base(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(config["dense_units"], activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(config["dropout"])(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="fruit_vgg16"), base


def enable_fine_tuning(base, block="block5"):
    """Train the named VGG block; VGG16 has no BatchNormalization layers."""
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(block + "_")
    if not any(layer.trainable and layer.weights for layer in base.layers):
        raise ValueError(f"No trainable weights found for VGG block: {block}")
