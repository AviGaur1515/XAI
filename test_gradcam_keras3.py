import os
import numpy as np
import tensorflow as tf
from utils.preprocessing import prepare_mri_inputs

model_path = "model/efficientnet_final.keras"
model = tf.keras.models.load_model(model_path, compile=False)
sample_path = "sample_data/sample_glioma.png"
mri_data = prepare_mri_inputs(sample_path, model_name="efficientnetb0")
img_array = mri_data["processed_batch"]
img_tensor = tf.cast(img_array, tf.float32)

print("Testing Keras 3 Grad-CAM pipeline...")

# Find base model (efficientnetb0)
base_model = None
for layer in model.layers:
    if hasattr(layer, "layers") and len(layer.layers) > 5:
        base_model = layer
        break

if base_model is None:
    base_model = model

print("Base model name:", base_model.name)

# Target layer inside base_model
target_layer_name = "top_conv"
target_layer = base_model.get_layer(target_layer_name)

# 1. Create sub-model from base_model inputs to target_layer output AND base_model output
# But wait! In base_model, top_conv is before top_bn and top_activation.
# Let's inspect base_model layers from top_conv to the end of base_model:
top_conv_idx = None
for idx, l in enumerate(base_model.layers):
    if l.name == target_layer_name:
        top_conv_idx = idx
        break

print(f"top_conv is layer index {top_conv_idx} of {len(base_model.layers)} in base_model")

# Create a sub_model that maps base_model.inputs -> target_layer.output
conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=target_layer.output)

# Remaining layers of base_model after top_conv
base_remaining_layers = base_model.layers[top_conv_idx + 1:]
print("Base remaining layers:", [l.name for l in base_remaining_layers])

# Classifier head layers (all layers in model after base_model)
head_layers = []
for layer in model.layers:
    if not isinstance(layer, tf.keras.layers.InputLayer) and layer.name != base_model.name:
        head_layers.append(layer)

print("Head layers:", [l.name for l in head_layers])

with tf.GradientTape() as tape:
    # 1. Forward pass up to top_conv
    conv_outputs = conv_model(img_tensor, training=False)
    tape.watch(conv_outputs)
    
    # 2. Forward pass through remaining base_model layers
    feat = conv_outputs
    for layer in base_remaining_layers:
        feat = layer(feat, training=False)
        
    # 3. Forward pass through classifier head layers
    for layer in head_layers:
        feat = layer(feat, training=False)
        
    predictions = feat
    pred_index = int(tf.argmax(predictions[0]).numpy())
    class_score = predictions[:, pred_index]

print("Predictions:", predictions.numpy())
print("Pred index:", pred_index)
print("Conv outputs shape:", conv_outputs.shape)

grads = tape.gradient(class_score, conv_outputs)
print("Grads is None?:", grads is None)
if grads is not None:
    print("Grads shape:", grads.shape)
    print("Grads min, max, mean:", grads.numpy().min(), grads.numpy().max(), grads.numpy().mean())
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.nn.relu(cam)
    cam_np = cam.numpy()
    mn, mx = cam_np.min(), cam_np.max()
    if (mx - mn) > 1e-8:
        cam_np = (cam_np - mn) / (mx - mn)
    else:
        cam_np = np.zeros_like(cam_np)
    print("CAM heatmap min, max, mean:", cam_np.min(), cam_np.max(), cam_np.mean())
    print("SUCCESS! GRAD-CAM WORKED PERFECTLY!")
