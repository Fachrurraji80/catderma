import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Deteksi Penyakit Kulit Kucing",
    page_icon="🐱"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "mobilenetv2_cat_skin_disease.keras"
    )

model = load_model()

with open("labels.json", "r") as f:
    class_names = json.load(f)

def predict_image(image):

    image = image.resize((224,224))

    img = np.array(image)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    idx = np.argmax(prediction)

    confidence = np.max(prediction)

    return class_names[idx], confidence

st.title("🐱 Klasifikasi Penyakit Kulit Kucing")

uploaded = st.file_uploader(
    "Upload gambar",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image)

    if st.button("Prediksi"):

        label, conf = predict_image(image)

        st.success(
            f"Hasil: {label}"
        )

        st.info(
            f"Confidence: {conf*100:.2f}%"
        )