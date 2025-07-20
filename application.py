import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ✅ Load the trained model
MODEL_PATH = "C:\\Users\\Admin\\Desktop\\internship_projects\\cats dogs classifeir\\saved_models\\cat_dog_classifier_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))          # Resize to 150x150
    image = image.convert('RGB')              # Ensure 3 channels
    img_array = np.array(image)
    img_array = img_array.reshape(1, 150, 150, 3)  # Now safe to reshape
    img_array = img_array / 255.0             # Normalize if needed
    return img_array
# ✅ Streamlit UI
st.set_page_config(page_title="Cat vs Dog Classifier")
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        
        if prediction > 0.5:
            st.success("It's a **Dog** 🐶")
        else:
            st.success("It's a **Cat** 🐱")
