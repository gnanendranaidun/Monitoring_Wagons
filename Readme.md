readme_content = """
# 🚆 Wagon Detection App (YOLOv8 + Streamlit)

A unified Streamlit-based web app that allows users to:
- 📤 Upload **images** for detection using a YOLOv8 `best.pt` model
- 📤 Upload **videos** for wagon detection using a custom motion+ROI-based algorithm
- 🖼️ View previously processed images and videos

---

## 📸 Features

### 🖼️ Image Detection
- Upload an image (`.jpg`, `.png`)
- Automatically runs a YOLOv8 trained model (`best.pt`)
- Annotated results are saved and displayed in a gallery

### 🎥 Video Detection
- Upload a `.mp4` or `.avi` video
- Performs wagon counting via background subtraction + line crossing logic
- Final processed video is saved and displayed

---

## 🗂️ Folder Structure

\`\`\`
project_root/
├── app.py                      # Main Streamlit app with tabs
├── wagon_detector.py           # Video wagon detection logic
├── weights/
│   └── best.pt                 # YOLOv8 trained model
├── uploads/                   # Uploaded images
├── videos/                    # Uploaded & processed videos
├── runs/
│   └── detect/
│       └── predict/           # YOLO image predictions
├── requirements.txt
├── README.md
\`\`\`

---

## 🚀 Getting Started

### 1. Clone the Repo

\`\`\`bash
git clone https://github.com/your-username/wagon-detection-app.git
cd wagon-detection-app
\`\`\`

### 2. Install Requirements

We recommend using a virtual environment.

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Place Your Model

Put your trained YOLOv8 model \`best.pt\` inside the \`weights/\` folder.

If you don’t have it yet, you can train one using:

\`\`\`bash
yolo detect train data=your_data.yaml model=yolov8n.pt epochs=50
\`\`\`

### 4. Run the App

\`\`\`bash
streamlit run app.py
\`\`\`

Then open the app in your browser:  
📍 [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Configuration

- \`wagon_detector.py\` handles custom logic for video-based detection using OpenCV
- YOLOv8 uses \`ultralytics\` and saves images to \`runs/detect/predict/\` by default (controlled using \`project\` and \`name\` arguments)

---

## 📦 requirements.txt

\`\`\`txt
streamlit
ultralytics
opencv-python
numpy
Pillow
\`\`\`

Install via:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 📸 Sample UI

- Tab 1: Upload and detect images
- Tab 2: Upload and process videos
- Each tab shows previously processed media automatically

---

## 🧠 Future Improvements

- Display detection class names and confidence scores
- Add download buttons for results
- Store results in a database for tracking

---

## 🛠️ Author

Built by @gnanendranaidun and @aditya-ranjan1234

---

## 📄 License

MIT License — feel free to use and modify!
"""

with open("README.md", "w") as f:
    f.write(readme_content)
