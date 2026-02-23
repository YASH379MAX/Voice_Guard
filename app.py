from flask import Flask, render_template, request, jsonify
import joblib
import librosa
import numpy as np
import os

app = Flask(__name__)

# ===== MODEL LOAD =====
MODEL = joblib.load("voice_model.pkl")
SCALER = joblib.load("scaler.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===== FEATURE EXTRACTION =====
def extract_features(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1).reshape(1, -1)


# ===== FRONTEND =====
@app.route("/")
def home():
    return render_template("index.html")


# ===== PREDICT API =====
@app.route("/predict", methods=["POST"])
def predict():

    # ✅ safety check
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # feature extraction
    features = extract_features(filepath)
    features = SCALER.transform(features)

    prediction = MODEL.predict(features)[0]
    probability = MODEL.predict_proba(features)[0]

    confidence = float(max(probability) * 100)

    result = "AI Voice" if prediction == 1 else "Human Voice"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)