"""Grad-CAM for the skin cancer CNN.

Given a trained Keras classifier and an input image, produce a heatmap that
highlights the regions most responsible for the predicted class. The approach
follows Selvaraju et al. (2017): weight the last conv layer's feature maps by
the gradient of the target class w.r.t. that layer, then ReLU and normalize.
"""
from io import BytesIO

import numpy as np
import tensorflow as tf
from matplotlib import cm
from PIL import Image

IMG_SIZE = 224


def _find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def preprocess_pil(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="float32")
    return img, np.expand_dims(arr, axis=0)


def _build_grad_model(model, last_conv_layer_name):
    # Sequential models loaded from disk don't have model.output until called once.
    # Rebuild a functional graph explicitly from a fresh Input through the existing layers.
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_out = x
    if conv_out is None:
        raise ValueError(f"Layer {last_conv_layer_name!r} not found in model")
    return tf.keras.models.Model(inputs=inputs, outputs=[conv_out, x])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer_name(model)
    grad_model = _build_grad_model(model, last_conv_layer_name)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        # Binary sigmoid: class score = preds[:, 0] for "malignant"
        class_score = preds[:, 0]
    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(preds[0, 0])


def overlay_heatmap(pil_image, heatmap, alpha=0.45):
    base = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    )
    colored = cm.get_cmap("jet")(heatmap_resized / 255.0)[:, :, :3]
    colored = (colored * 255).astype("uint8")
    blended = (np.array(base) * (1 - alpha) + colored * alpha).astype("uint8")
    return Image.fromarray(blended)


def pil_to_base64_jpeg(pil_image):
    import base64
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
