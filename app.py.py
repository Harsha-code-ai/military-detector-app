import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("my_model.h5")

st.title("Military Equipment Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] >= 0.5:
        st.success("✅ This is REAL equipment.")
    else:
        st.error("❌ This is FAKE / Decoy equipment.")
