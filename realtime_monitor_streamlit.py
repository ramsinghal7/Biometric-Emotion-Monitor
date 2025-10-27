import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import joblib
import math
import time

# -------------------------------------------------
# 🧠 Streamlit Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Industrial Emotion & Behavior Monitor", layout="centered")
st.title("🎥 AI-Based Real-Time Emotion & Behavior Monitor")

st.markdown("""
Use this dashboard to monitor worker emotions and fatigue in industrial environments.  
🟢 **Live Camera Feed:** Always on  
📸 **Capture & Analyze Button:** Takes snapshot, runs emotion + fatigue detection  
⚠️ **Alerts:** Triggered when fatigue or stress is detected  
""")

# -------------------------------------------------
# 🧩 Load Trained Model and Label Encoder
# -------------------------------------------------
try:
    clf = joblib.load('rf_model.joblib')
    le = joblib.load('label_encoder.joblib')
    if isinstance(le, list):
        le = le[0]
except Exception as e:
    st.error(f"❌ Model or encoder load error: {e}")
    st.stop()

# -------------------------------------------------
# 🎯 Mediapipe Setup
# -------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
INNER_MOUTH = [13,14,78,308]

# -------------------------------------------------
# 📷 Camera Initialization
# -------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("⚠ Webcam not accessible. Please check permissions or camera drivers.")
    st.stop()

FRAME_WINDOW = st.image([])
capture_btn = st.button("📸 Capture & Analyze")
status_text = st.empty()

# -------------------------------------------------
# 🔁 Continuous Live Preview Loop
# -------------------------------------------------
st.info("Live camera feed started. Click '📸 Capture & Analyze' anytime to process a frame.")
st.markdown("---")

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not read from camera.")
        break

    # Show the live video feed
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # If capture button pressed, analyze current frame
    if capture_btn:
        st.subheader("🧠 Captured Frame Analysis")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------------------------------------------
        # 🧩 Emotion Detection (DeepFace)
        # --------------------------------------------
        try:
            analysis = DeepFace.analyze(img_path=rgb, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion = analysis.get('dominant_emotion', 'neutral')
            emotion_score = max(analysis.get('emotion', {}).values()) if analysis.get('emotion') else 0
        except Exception:
            emotion, emotion_score = 'neutral', 0

        # --------------------------------------------
        # 🧩 Facial Landmarks via Mediapipe
        # --------------------------------------------
        results = face_mesh.process(rgb)
        ear, mar, head_angle = 0, 0, 0
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

            # EAR
            le_pts = [pts[i] for i in LEFT_EYE]
            re_pts = [pts[i] for i in RIGHT_EYE]
            left_ear = (euclid(le_pts[1], le_pts[5]) + euclid(le_pts[2], le_pts[4])) / (2.0 * euclid(le_pts[0], le_pts[3]) + 1e-6)
            right_ear = (euclid(re_pts[1], re_pts[5]) + euclid(re_pts[2], re_pts[4])) / (2.0 * euclid(re_pts[0], re_pts[3]) + 1e-6)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mouth_pts = [pts[i] for i in INNER_MOUTH]
            mar = euclid(mouth_pts[0], mouth_pts[1]) / (euclid(mouth_pts[2], mouth_pts[3]) + 1e-6)

            # Head Angle
            left_eye_center = np.mean([pts[i] for i in [33,133]], axis=0)
            right_eye_center = np.mean([pts[i] for i in [362,263]], axis=0)
            nose_tip = pts[1]
            eye_center = (left_eye_center + right_eye_center) / 2
            vec = np.array(nose_tip) - eye_center
            head_angle = math.degrees(math.atan2(vec[1], vec[0]))

        # --------------------------------------------
        # 🤖 Prediction Using ML Model
        # --------------------------------------------
        features = np.array([[emotion_score, ear, mar, head_angle]])
        try:
            pred_idx = clf.predict(features)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
        except Exception:
            pred_label = "unknown"

        pred_prob = np.max(clf.predict_proba(features))

        # --------------------------------------------
        # 🖼️ Display Captured Image + Analysis
        # --------------------------------------------
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Frame", use_column_width=True)

        st.write(f"**Dominant Emotion:** {emotion}")
        st.write(f"**EAR:** {ear:.3f} | **MAR:** {mar:.3f} | **Head Angle:** {head_angle:.1f}")
        st.write(f"**Model Prediction:** {pred_label} ({pred_prob:.2f})")

        # Alert Logic
        if pred_label in ['fatigued', 'stressed'] and pred_prob > 0.6:
            st.error("⚠ ALERT: Worker may be Fatigued or Stressed — suggest taking a break!")
        else:
            st.success("✅ Status: Focused / Normal")

        # Reset button so it doesn't trigger repeatedly
        capture_btn = False

    # Add small delay for frame refresh
    time.sleep(0.1)

cap.release()
