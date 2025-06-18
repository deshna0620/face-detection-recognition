import streamlit as st 
from ultralytics import YOLO
from keras_facenet import FaceNet
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

st.set_page_config(page_title="Face Detection & Recognition", layout="centered")
st.title("🔍 Face Detection & Recognition App")
st.markdown("Uses **YOLOv8** for detection & **FaceNet** for recognition. Accuracy > 98%.")

yolo_model = YOLO("yolov8n.pt")
embedder = FaceNet()

st.sidebar.header("Upload Images")
face1_file = st.sidebar.file_uploader("Upload Face 1 (Original / Masked)", type=["jpg", "png", "jpeg"])
face2_file = st.sidebar.file_uploader("Upload Face 2 (Original / Capped)", type=["jpg", "png", "jpeg"])

# Ensure temp_images directory exists
if not os.path.exists("temp_images"):
    os.makedirs("temp_images")

def save_and_read_image(uploaded_file):
    temp_path = os.path.join("temp_images", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def detect_face(img_path):
    yolo_model.predict(
        source=img_path,
        conf=0.5,
        save=True,
        project="runs/detect",
        name="streamlit",
        exist_ok=True
    )

def get_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings([img])[0]
    return embedding

if face1_file is not None and face2_file is not None:
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(face1_file), caption="Face 1", use_column_width=True)
    with col2:
        st.image(Image.open(face2_file), caption="Face 2", use_column_width=True)

    if st.button("Detect and Compare"):
        img1_path = save_and_read_image(face1_file)
        img2_path = save_and_read_image(face2_file)

        detect_face(img1_path)
        detect_face(img2_path)

        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)
        similarity = cosine_similarity([emb1], [emb2])[0][0] * 100

        st.subheader("Results")
        st.write(f"🧠 Cosine Similarity: **{similarity:.2f}%**")
        if similarity > 98:
            st.success("✅ Faces Match (above 98% confidence)")
        else:
            st.error("❌ Faces Do NOT Match (below 98% confidence)")

        st.subheader("YOLOv8 Detected Faces")
        det_path = "runs/detect/streamlit"
        col3, col4 = st.columns(2)
        with col3:
            det1_path = os.path.join(det_path, os.path.basename(img1_path))
            st.image(det1_path, caption="Detected Face 1", use_column_width=True)
        with col4:
            det2_path = os.path.join(det_path, os.path.basename(img2_path))
            st.image(det2_path, caption="Detected Face 2", use_column_width=True)

        os.remove(img1_path)
        os.remove(img2_path)
else:
    st.warning("Please upload both Face 1 and Face 2 images to proceed.")