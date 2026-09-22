import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load ML model
with open("photo_quality_ml.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI-Based Photo Quality Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((64, 64))
    img_array = np.array(img).flatten() / 255.0
    img_array = img_array.reshape(1, -1)

    # Prediction
    prediction = model.predict(img_array)[0]

    if prediction == 1:
        st.success("Photo Quality: High Quality")
    else:
        st.warning("Photo Quality: Low Quality")
