# 🧠 AI-Based Real-Time Emotion & Behavior Monitor

A computer vision–based system that detects **emotions, fatigue, and attention levels** of workers in real time using webcam input.  
This project combines **DeepFace** for emotion recognition, **Mediapipe** for facial landmark tracking, and a **Random Forest classifier** for behavioral analysis — all wrapped in a live **Streamlit dashboard**.

---

## 🚀 Features

- 🎥 **Continuous Live Camera Feed** – Always-on webcam preview  
- 📸 **Capture & Analyze Button** – Takes snapshot and runs prediction instantly  
- 🧠 **Emotion Detection** – Powered by DeepFace  
- 👁️ **Behavioral Biometrics** – Uses Mediapipe to extract EAR, MAR, and head angle  
- ⚡ **ML-Based Classification** – Predicts worker state: *Focused*, *Fatigued*, or *Stressed*  
- 🖥️ **Streamlit Dashboard** – Interactive and professional front-end interface  

---

## 🏗️ Project Structure

Biometric-Emotion-Monitor/
│
├── train_model.py # Train and save the ML model
├── realtime_monitor_streamlit.py # Streamlit dashboard (real-time detection)
├── rf_model.joblib # Trained RandomForest model
├── label_encoder.joblib # Saved label encoder
├── requirements.txt # Python dependencies
├── features.csv # Optional dataset (for model training)
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files

## 🧩 Tech Stack

| Component | Library | Description |
|------------|----------|-------------|
| Emotion Detection | `DeepFace` | Detects facial emotion from frame |
| Landmark Tracking | `Mediapipe` | Extracts face mesh landmarks |
| ML Model | `scikit-learn` | RandomForestClassifier for classification |
| Dashboard | `Streamlit` | Real-time visualization and user interaction |
| Image Processing | `OpenCV` | Webcam capture and frame analysis |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Biometric-Emotion-Monitor.git
cd Biometric-Emotion-Monitor

python -m venv biometrics_env
biometrics_env\Scripts\activate   # Windows
# or
source biometrics_env/bin/activate   # macOS / Linux

pip install -r requirements.txt

streamlit run realtime_monitor_streamlit.py

The webcam feed is continuously processed.

DeepFace identifies the dominant emotion.

Mediapipe extracts facial metrics:

EAR (Eye Aspect Ratio) – to detect drowsiness

MAR (Mouth Aspect Ratio) – to detect yawning

Head Angle – to monitor attention

These features + emotion score are fed into a trained RandomForest model.

The model classifies the worker's current state as:

🟢 Focused

🟡 Fatigued

🔴 Stressed

Results and alerts are displayed live on the Streamlit dashboard.
