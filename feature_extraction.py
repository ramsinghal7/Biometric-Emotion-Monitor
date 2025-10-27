# feature_extraction.py
import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from deepface import DeepFace
import math
from imutils import face_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

DATA_DIR = 'data_raw'
OUT_CSV = 'features.csv'

# eye and mouth landmark indices for Mediapipe face mesh
# using indices from MediaPipe Face Mesh (468 points) - these are approximate
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # outer, upper, lower etc
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
INNER_MOUTH = [13, 14, 78, 308]  # approximate top, bottom, left, right

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def aspect_ratio(pts):
    # expects list of (x,y)
    # simple MAR: vertical_dist / horizontal_dist
    vertical = euclid(pts[0], pts[1])
    horizontal = euclid(pts[2], pts[3])
    if horizontal == 0: 
        return 0
    return vertical / horizontal

rows = []
for label in os.listdir(DATA_DIR):
    labdir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(labdir): continue
    for fname in os.listdir(labdir):
        path = os.path.join(labdir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # DeepFace emotion (may be slow)
        try:
            analysis = DeepFace.analyze(img_path = rgb, actions=['emotion'], enforce_detection=False)
            # DeepFace returns dict or list depending on version
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotion = analysis.get('dominant_emotion', 'neutral')
            emotion_score = max(analysis.get('emotion', {}).values()) if analysis.get('emotion') else 0
        except Exception as e:
            emotion = 'neutral'
            emotion_score = 0

        # Mediapipe landmarks
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            # no face — skip for now
            ear = 0
            mar = 0
            head_angle = 0
        else:
            lm = results.multi_face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
            # Left eye
            le = [pts[i] for i in LEFT_EYE]
            re = [pts[i] for i in RIGHT_EYE]
            # simple EAR-like: vertical/horizontal for left and right average
            left_ear = (euclid(le[1], le[5]) + euclid(le[2], le[4])) / (2.0 * euclid(le[0], le[3]) + 1e-6)
            right_ear = (euclid(re[1], re[5]) + euclid(re[2], re[4])) / (2.0 * euclid(re[0], re[3]) + 1e-6)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mouth_pts = [pts[i] for i in INNER_MOUTH]
            mar = aspect_ratio(mouth_pts)

            # head pose approximation: use points to estimate yaw by nose tip vs eyes center
            left_eye_center = np.mean([pts[i] for i in [33,133]], axis=0)
            right_eye_center = np.mean([pts[i] for i in [362,263]], axis=0)
            nose_tip = pts[1] if len(pts) > 1 else pts[4]
            # vector eye -> nose
            eye_center = (left_eye_center + right_eye_center) / 2
            vec = nose_tip - eye_center
            head_angle = math.degrees(math.atan2(vec[1], vec[0]))  # rough orientation

        rows.append({
            'filename': path,
            'label': label,
            'emotion': emotion,
            'emotion_score': float(emotion_score),
            'ear': float(ear),
            'mar': float(mar),
            'head_angle': float(head_angle)
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Saved features to", OUT_CSV)
