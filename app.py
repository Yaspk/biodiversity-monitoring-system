import os
import logging
import random
import base64
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "biodiversity_model.h5"
IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["elephant", "lion", "panda", "zebra"]
LOG_FILE = "biodiversity_log.csv"

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------
# APP INIT
# -----------------------------
app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)
logging.info("Model loaded")

# -----------------------------
# HELPERS
# -----------------------------
def get_environmental_data():
    return {
        "temperature": round(random.uniform(15, 40), 2),
        "humidity": round(random.uniform(30, 90), 2),
        "vegetation_index": round(random.uniform(0.2, 0.9), 2)
    }

def preprocess_image_from_bytes(image_bytes):
    img = image.load_img(BytesIO(image_bytes), target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def log_detection(species, confidence, zone):
    exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a") as f:
        if not exists:
            f.write("timestamp,species,confidence,zone\n")

        f.write(
            f"{datetime.now().isoformat()},"
            f"{species},"
            f"{round(confidence * 100, 2)},"
            f"{zone}\n"
        )

    logging.info(f"Logged: {species} in Zone {zone}")

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate file
        if "file" not in request.files:
            return render_template("result.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("result.html", error="No file selected")

        # Validate zone
        zone_selected = request.form.get("zone")
        if not zone_selected:
            return render_template("result.html", error="Please select a zone")

        # Read image
        image_bytes = file.read()

        # Preprocess
        img_array = preprocess_image_from_bytes(image_bytes)

        # Predict
        preds = model.predict(img_array)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])

        species = CLASS_NAMES[class_id]

        # Zone info
        zone_data = {
            "zone": zone_selected,
            "status": "User Assigned Zone",
            "color": "green" if zone_selected == "A"
                     else "orange" if zone_selected == "B"
                     else "red"
        }

        # Environmental simulation
        env_data = get_environmental_data()

        # Log
        log_detection(species, confidence, zone_selected)

        # Image to base64
        encoded_img = base64.b64encode(image_bytes).decode("utf-8")

        return render_template(
            "result.html",
            species=species.capitalize(),
            confidence=round(confidence * 100, 2),
            zone=zone_data,
            environment=env_data,
            image_data=encoded_img,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        logging.exception("Prediction failed")
        return render_template("result.html", error=str(e))

@app.route("/map")
def map_dashboard():
    zones = {
        "A": {},
        "B": {},
        "C": {}
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)

        for z in ["A", "B", "C"]:
            zone_df = df[df["zone"] == z]
            zones[z] = zone_df["species"].value_counts().to_dict()

    return render_template("map.html", zones=zones)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

