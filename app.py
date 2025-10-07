from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "efficientnetb3_kneeKL.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['0', '1', '2', '3', '4']
IMG_SIZE = (300, 300)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("xray")
        if file:
            img = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
            arr = np.expand_dims(np.array(img), axis=0).astype("float32")
            preds = model.predict(arr)[0]
            pred_idx = int(np.argmax(preds))
            grade = class_names[pred_idx]
            conf = float(preds[pred_idx])
            result = f"K-L Grade: {grade} (Confidence: {conf:.2f})"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
