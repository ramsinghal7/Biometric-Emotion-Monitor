# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load extracted features
df = pd.read_csv('features.csv')

# Select features and target
X = df[['emotion_score', 'ear', 'mar', 'head_angle']].fillna(0)
y = df['label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and label encoder properly
joblib.dump(clf, 'rf_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("✅ Model saved as 'rf_model.joblib'")
print("✅ Label encoder saved as 'label_encoder.joblib'")
