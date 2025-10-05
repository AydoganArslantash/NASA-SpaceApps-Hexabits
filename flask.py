from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import json

model = tf.keras.models.load_model("the_model.keras")
scaler = joblib.load("scaler.pkl")
with open("feature_names.json") as f:
    feature_names = json.load(f)

app = Flask(_name_)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([data[feat] for feat in feature_names]).reshape(1, -1)
    
    
    X_scaled = scaler.transform(X)

    pred_probs = model.predict(X_scaled)
    pred_class = int(np.argmax(pred_probs, axis=1)[0])

    return jsonify({"predicted_class": pred_class, "probabilities": pred_probs[0].tolist()})

if _name_ == "_main_":
    app.run(debug=True)
