import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import uuid
from wagon_detector import process_video

# --- Config ---
VIDEO_FOLDER = "videos"
IMAGE_UPLOAD_FOLDER = "uploads"
IMAGE_OUTPUT_FOLDER = "runs/detect/predict"
MODEL_PATH = "model/best.pt"

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model for images
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Wagon Detection Suite", layout="centered")
st.title("🚆 Wagon Detection App")

# ----------------------------------
# 🧭 Tabs
# ----------------------------------
tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

# ----------------------------------
# 🖼️ Image Detection Tab
# ----------------------------------
with tab1:
    st.subheader("📁 Detected Images")

    if os.path.exists(IMAGE_OUTPUT_FOLDER):
        image_files = [
            f for f in os.listdir(IMAGE_OUTPUT_FOLDER)
            if f.lower().endswith((".jpg", ".png"))
        ]
        if image_files:
            for img in image_files:
                st.image(os.path.join(IMAGE_OUTPUT_FOLDER, img), caption=img, use_column_width=True)
        else:
            st.info("No detections yet.")
    else:
        st.info("No detections yet.")

    st.subheader("📤 Upload Image")
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_img:
        unique_filename = f"{uuid.uuid4().hex}_{uploaded_img.name}"
        img_path = os.path.join(IMAGE_UPLOAD_FOLDER, unique_filename)

        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())

        st.success(f"Uploaded: {uploaded_img.name}")
        st.image(img_path, caption="Original Image", use_column_width=True)

        st.info("🔍 Running YOLO detection...")
        model(
            img_path,
            save=True,
            project="runs/detect",
            name="predict",
            exist_ok=True
        )


        result_img_path = os.path.join(IMAGE_OUTPUT_FOLDER, os.path.basename(img_path))
        if os.path.exists(result_img_path):
            st.success("✅ Detection complete!")
            st.image(result_img_path, caption="Detected Image", use_column_width=True)
        else:
            st.error("❌ Failed to find result.")

# ----------------------------------
# 🎥 Video Detection Tab
# ----------------------------------
with tab2:
    st.subheader("📁 Processed Videos")

    video_files = sorted([
        f for f in os.listdir(VIDEO_FOLDER)
        if f.startswith("processed_") and f.endswith((".mp4", ".avi"))
    ])

    if video_files:
        for vid in video_files:
            st.write(f"🎞️ {vid}")
            with open(os.path.join(VIDEO_FOLDER, vid), "rb") as vfile:
                st.video(vfile.read())
    else:
        st.info("No processed videos yet.")

    st.subheader("📤 Upload Video")
    uploaded_vid = st.file_uploader("Upload a .mp4 or .avi video", type=["mp4", "avi"], key="video_uploader")

    if uploaded_vid:
        raw_video_path = os.path.join(VIDEO_FOLDER, uploaded_vid.name)
        with open(raw_video_path, "wb") as f:
            f.write(uploaded_vid.read())

        st.success(f"Uploaded: {uploaded_vid.name}")

        output_path = os.path.join(VIDEO_FOLDER, f"processed_{uploaded_vid.name}")
        st.info("⏳ Processing video... Please wait.")
        wagon_count = process_video(raw_video_path, output_path, display=False)
        st.success(f"✅ Processing done! Wagons detected: {wagon_count}")

        with open(output_path, "rb") as out_vid:
            st.video(out_vid.read())
