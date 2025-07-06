import streamlit as st
import os
from io import BytesIO
from wagon_detector import process_video

# Folder to store uploaded and processed videos
VIDEO_DB = "videos"
os.makedirs(VIDEO_DB, exist_ok=True)

st.set_page_config(page_title="Wagon Detector", layout="centered")
st.title("🚆 Wagon Detection System")

# -------------------------------
# Show Processed Videos Library first
# -------------------------------
st.subheader("📁 Processed Videos Library")

video_files = sorted([f for f in os.listdir(VIDEO_DB) if f.startswith("processed_") and f.endswith((".mp4", ".avi"))])

if video_files:
    for vid in video_files:
        vid_path = os.path.join(VIDEO_DB, vid)
        st.write(f"🎞️ {vid}")
        with open(vid_path, "rb") as vfile:
            st.video(vfile.read())
else:
    st.info("No processed videos yet. Upload one below to get started!")

# -------------------------------
# Upload new video
# -------------------------------
st.subheader("📤 Upload a new video")
uploaded_file = st.file_uploader("Upload .mp4 or .avi file", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded file to disk
    raw_video_path = os.path.join(VIDEO_DB, uploaded_file.name)
    with open(raw_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Process and store output video
    output_path = os.path.join(VIDEO_DB, f"processed_{uploaded_file.name}")
    st.info("⏳ Processing video... This may take a while.")
    wagon_count = process_video(raw_video_path, output_path, display=False)
    st.success(f"🎉 Processing done! Wagons detected: {wagon_count}")

    # Display processed video immediately
    with open(output_path, "rb") as out_vid:
        st.video(out_vid.read())
