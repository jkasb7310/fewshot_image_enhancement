import tempfile
import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import keras
from keras import layers

import tensorflow as tf

import keras
import tensorflow as tf
import streamlit as st
import shutil

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

# Load the SavedModel
Mirnet_model = tf.keras.models.load_model('finetune_mirnet', custom_objects={'peak_signal_noise_ratio': peak_signal_noise_ratio, 'charbonnier_loss': charbonnier_loss})

# Verify the loaded model
# Mirnet_model.summary()


# Preprocessing function
def preprocess_large_image(image, max_dimension=1024):
    original_height, original_width = image.size
    scale_factor = min(max_dimension / original_height, max_dimension / original_width)
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    resized_image = image.resize((new_width, new_height))

    # Pad to ensure divisibility by 2^3 (or your model's factor)
    factor = 2 ** 3
    padded_height = (new_height // factor) * factor
    padded_width = (new_width // factor) * factor
    padded_image = resized_image.crop((0, 0, padded_width, padded_height))

    return np.array(padded_image), (original_height, original_width), (new_height, new_width)

# Inference function
def infer_large_image(image, original_shape, resized_shape):
    # Add batch dimension
    input_image = np.expand_dims(image, axis=0).astype("float32") / 255.0

    # Model prediction
    output = Mirnet_model.predict(input_image, verbose=0)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255).astype(np.uint8)

    # Resize back to original dimensions
    #output_image = Image.fromarray(output_image).resize(original_shape[::-1], Image.BICUBIC)
    output_image = Image.fromarray(output_image).resize((512,512), Image.Resampling.LANCZOS)
    return output_image


st.title("Low-Light Image Enhancement")
st.write("Upload an image to enhance it using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    input_image = Image.open(uploaded_file).convert("RGB")
    print(uploaded_file, type(uploaded_file))

    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    st.write("Enhancing...")
    preprocessed_image, original_shape, resized_shape = preprocess_large_image(input_image)
    enhanced_image = infer_large_image(preprocessed_image, original_shape, resized_shape)
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Save the enhanced image as a temporary file
    enhanced_image.save(f'enhanced{file_extension}')

    # Run the Real-ESRGAN enhancement script
    os.system(f'python inference_realesrgan.py -n net_g_latest -i enhanced{file_extension} -o ./')
     
    final_img = Image.open(f'enhanced_out{file_extension}').convert("RGB")
    # Display the results
    st.image(final_img, caption="Enhanced Image", use_column_width=True)

    # Option to download the enhanced image
    st.download_button(
        label="Download Enhanced Image",
        data=enhanced_image.tobytes(),
        file_name=f"enhanced_out{file_extension}",
        mime=f"image/{file_extension}"
    )

