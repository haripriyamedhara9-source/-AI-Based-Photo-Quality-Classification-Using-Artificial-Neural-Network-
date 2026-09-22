import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("photo_quality_ann.h5")

st.title("AI-Based Photo Quality Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((64, 64))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    prediction = model.predict(img_array)

    if prediction[0][0] >= 0.5:
        st.success("Photo Quality: High Quality")
    else:
        st.warning("Photo Quality: Low Quality")
