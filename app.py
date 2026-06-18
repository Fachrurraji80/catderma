import streamlit as st
import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(
    page_title="Deteksi Penyakit Kulit Kucing",
    page_icon="🐱"
)

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(
        model_path="cat_skin_disease.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.json", "r") as f:
    class_names = json.load(f)

def preprocess(img):
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return img

def predict_image(image):

    image = image.resize((224, 224))

    img = np.array(image)

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = preprocess(img)

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[idx], confidence

st.title("🐱 Klasifikasi Penyakit Kulit Kucing")

uploaded = st.file_uploader(
    "Upload gambar kucing",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image, caption="Gambar yang diupload")

    if st.button("Prediksi"):

        label, conf = predict_image(image)

        st.success(f"Hasil Prediksi: {label}")
        st.info(f"Tingkat Keyakinan: {conf*100:.2f}%")
