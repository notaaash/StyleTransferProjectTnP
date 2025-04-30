import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return tf.constant(img, dtype=tf.float32)[tf.newaxis, ...]

st.title("🎨 Neural Style Transfer Web App")

content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
alpha = st.slider("Style Strength", 0.0, 1.0, 0.5)

if content_image and style_image:
    content_tensor = load_image(content_image)
    style_tensor = load_image(style_image)
    stylized_image = model(content_tensor, style_tensor)[0]
    blended_image = alpha * stylized_image + (1 - alpha) * content_tensor
    blended_image = tf.clip_by_value(blended_image, 0.0, 1.0)

    st.subheader("Result")
    st.image(blended_image.numpy()[0])
