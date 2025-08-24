import streamlit as st
from ultralytics import YOLO
import tempfile, os
from PIL import Image
import cv2
import numpy as np
import time

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🤖", layout="wide")

st.sidebar.title("⚙ Settings")
default_model = "yolov8n.pt"
model_path = st.sidebar.text_input("Model path (.pt weights)", value=default_model)


st.title("YOLOv8 Object Detection")
st.caption("Upload image, video, or use webcam for real-time detection")

@st.cache_resource(show_spinner=False)
def load_model(weights_path):
    return YOLO(weights_path)

model = load_model(model_path)

mode = st.radio("Choose Mode", ["Image", "Video", "Webcam"], horizontal=True)


# ---------------- Image Mode ----------------
if mode == "Image":
    file = st.file_uploader("Upload Image", type=["jpg","jpeg","png","bmp","webp"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Run Detection"):
            results = model.predict(source=np.array(image), verbose=False)
            res = results[0].plot()
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detection Result", use_column_width=True)


# ---------------- Video Mode ----------------
elif mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])
    if file:
        tdir = tempfile.mkdtemp()
        in_path = os.path.join(tdir, file.name)
        with open(in_path, "wb") as f:
            f.write(file.read())
        st.video(in_path)

        if st.button("Run Detection on Video"):
            st.info("Processing video... please wait ⏳")
            results = model.predict(source=in_path, stream=True, verbose=False)
            thumbs = []
            frame_count = 0
            for r in results:
                plotted = r.plot()
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                if frame_count % 20 == 0:  
                    thumbs.append(plotted)
                frame_count += 1
            if thumbs:
                st.image(thumbs, caption=[f"Frame {i*20}" for i in range(len(thumbs))], use_column_width=True)


# ---------------- Webcam Mode ----------------
elif mode == "Webcam":
    run_webcam = st.checkbox("Start Webcam")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    iou = st.slider("IOU Threshold", 0.0, 1.0, 0.45)

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
            res = results[0].plot()
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            stframe.image(res, channels="RGB", use_column_width=True)
            time.sleep(0.03)
        cap.release()


