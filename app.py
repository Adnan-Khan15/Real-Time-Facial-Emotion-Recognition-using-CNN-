# app.py
import os
import time
import av
import cv2
import numpy as np
import streamlit as st
from collections import deque
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

st.set_page_config(page_title="Real-Time Facial Emotion Recognition", layout="wide")

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_DEFAULT_PATH = "models/emotion_best.keras"

# ------------------------ SIDEBAR ------------------------
st.sidebar.header("Built by Adnan Khan for DEEP LEARNING & REINFORCEMENT Learning Mini Project")
use_webcam = st.sidebar.checkbox("Use Webcam (Real-time)", True)
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.01)
smooth_frames = st.sidebar.slider("Smoothing Frames", 1, 15, 5)
model_path = st.sidebar.text_input("Model path", MODEL_DEFAULT_PATH)

# ------------------------ MODEL LOADING ------------------------
@st.cache_resource(show_spinner=True)
def load_emotion_model(path):
    try:
        model = load_model(path, compile=False)
        st.sidebar.success("Model loaded normally.")
        return model
    except Exception:
        # fallback if preprocessing layers exist
        from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast
        custom = {
            "RandomFlip": RandomFlip,
            "RandomRotation": RandomRotation,
            "RandomContrast": RandomContrast,
        }
        model = load_model(path, custom_objects=custom, compile=False)
        st.sidebar.warning("Model loaded with fallback.")
        return model

try:
    model = load_emotion_model(model_path)
except Exception as e:
    st.error(f"Model load failed: {e}")
    model = None

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# ------------------------ HELPERS ------------------------
def preprocess_face(face_img):
    if face_img is None or face_img.size == 0:
        return None
    face_resized = cv2.resize(face_img, (96, 96))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
    return batch


# ------------------------ VIDEO PROCESSOR ------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=smooth_frames)
        self.latest_vector = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) > 0 and model is not None:
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = faces[0]
            pad = int(0.15 * w)
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)

            face_crop = img[y1:y2, x1:x2]
            batch = preprocess_face(face_crop)

            if batch is not None:
                raw = model.predict(batch, verbose=0).reshape(-1)
                if len(raw) != len(CLASS_NAMES):
                    vec = np.zeros(len(CLASS_NAMES))
                    m = min(len(raw), len(vec))
                    vec[:m] = raw[:m]
                else:
                    vec = raw.astype(np.float32)

                self.buffer.append(vec)
                smooth = np.mean(self.buffer, axis=0)
                self.latest_vector = smooth.copy()

                label_idx = int(np.argmax(smooth))
                score = float(smooth[label_idx])
                label = CLASS_NAMES[label_idx]

                color = (0, 255, 0) if score >= threshold else (0, 200, 200)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img, f"{label} ({score:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2
                )
        else:
            self.latest_vector *= 0.9  # decay slowly

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------ LIVE UI ------------------------
st.title("Real-Time Facial Emotion Recognition using DEEP LEARNING")

video_col, chart_col = st.columns([2, 3])

with video_col:
    st.subheader("Live Feed")
    rtc_ctx = None
    if use_webcam:
        rtc_ctx = webrtc_streamer(
            key="emotion",
            video_processor_factory=EmotionProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    else:
        st.info("Webcam disabled.")

with chart_col:
    st.subheader("Emotion Confidence Levels")
    chart_placeholder = st.empty()

# ------------------------ REAL-TIME CHART LOOP ------------------------
if use_webcam and rtc_ctx and rtc_ctx.video_processor:

    processor = rtc_ctx.video_processor

    # streamlit "loop"
    while True:
        if processor is None:
            break

        vec = processor.latest_vector
        vec = np.clip(vec, 0, 1)

        fig = go.Figure(
            data=[go.Bar(x=CLASS_NAMES, y=vec, marker_color="lightskyblue")],
            layout=go.Layout(
                yaxis=dict(range=[0, 1]),
                template="plotly_dark",
                height=350
            )
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.15)  # ~7 FPS update

        # If user presses STOP
        if not rtc_ctx.state.playing:
            break

# ------------------------ IMAGE UPLOAD ------------------------
st.subheader("Upload an image for inference")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded and model is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True) 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(0.15*w)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
        crop = img[y1:y2, x1:x2]

        batch = preprocess_face(crop)
        raw = model.predict(batch, verbose=0).reshape(-1)

        if len(raw) != len(CLASS_NAMES):
            vec = np.zeros(len(CLASS_NAMES))
            m = min(len(raw), len(vec))
            vec[:m] = raw[:m]
        else:
            vec = raw

        idx = int(np.argmax(vec))
        st.success(f"Predicted: **{CLASS_NAMES[idx]}** ({vec[idx]:.2f})")

        # update chart with image prediction too
        chart = go.Figure(
            data=[go.Bar(x=CLASS_NAMES, y=vec, marker_color="lightskyblue")],
            layout=go.Layout(yaxis=dict(range=[0,1]), template="plotly_dark", height=350)
        )
        chart_placeholder.plotly_chart(chart, use_container_width=True)
