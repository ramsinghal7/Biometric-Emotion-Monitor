# capture_and_label.py
import cv2
import os
import time

# folders for labeled data
LABELS = ['focused', 'fatigued', 'stressed']
SAVE_DIR = 'data_raw'
os.makedirs(SAVE_DIR, exist_ok=True)
for lbl in LABELS:
    os.makedirs(os.path.join(SAVE_DIR, lbl), exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press F for focus, G for fatigued, H for stressed. Q to quit.")
count = {lbl: len(os.listdir(os.path.join(SAVE_DIR, lbl))) for lbl in LABELS}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, "F:Focused G:Fatigued H:Stressed Q:Quit", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Capture (press key to save frame)", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):
        path = os.path.join(SAVE_DIR, 'focused', f"f_{count['focused']}.jpg")
        cv2.imwrite(path, frame)
        count['focused'] += 1
        print("Saved", path)
    elif key == ord('g'):
        path = os.path.join(SAVE_DIR, 'fatigued', f"g_{count['fatigued']}.jpg")
        cv2.imwrite(path, frame)
        count['fatigued'] += 1
        print("Saved", path)
    elif key == ord('h'):
        path = os.path.join(SAVE_DIR, 'stressed', f"h_{count['stressed']}.jpg")
        cv2.imwrite(path, frame)
        count['stressed'] += 1
        print("Saved", path)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
