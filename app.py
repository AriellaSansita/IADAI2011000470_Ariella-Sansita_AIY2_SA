"""
ParkVision AI - Intelligent Urban Parking Analytics & Space Optimisation
--------------------------------------------------------------------------
Streamlit web app that:
  1. Lets a user upload a parking lot image
  2. Runs slot-level occupancy detection (classification-based approach,
     using a MobileNet/EfficientNet model trained on cropped slot images)
  3. Draws colour-coded overlays (green = empty, red = occupied)
  4. Computes total / occupied / available slots + utilisation %
  5. Classifies congestion level (Low / Moderate / High)
  6. Generates a simple recommendation
  7. Displays everything in a clean, responsive dashboard

HOW TO PLUG IN YOUR OWN TRAINED MODEL
--------------------------------------
This app expects two things that YOU produce in Steps 2-3 of the assignment:

1. A trained Keras model file, e.g. "parking_model.h5", trained to classify
   a cropped slot image as "empty" or "occupied" (from Step 3).
2. A "slots.json" file describing where each parking slot is in the image,
   as a list of bounding boxes: [{"id": 1, "x": 10, "y": 20, "w": 80, "h": 40}, ...]
   You can generate this once per camera view (e.g. using a simple
   annotation tool, or by eyeballing pixel coordinates on a sample image).

If you instead trained a YOLO model that detects slots directly (no
slots.json needed), see the "detect_slots_yolo()" function below and swap
it in for "detect_slots_classification()".

Run with:
    streamlit run app.py
"""

import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow/Keras is only needed for the classification-based approach.
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "parking_model.h5"     # your trained MobileNet/EfficientNet model
SLOTS_PATH = "slots.json"           # slot bounding-box definitions
IMG_SIZE = (224, 224)               # must match the size used during training

LOW_THRESHOLD = 40      # % occupancy below this  -> Low congestion
HIGH_THRESHOLD = 75     # % occupancy above this   -> High congestion

COLOR_EMPTY = (0, 200, 0)     # green (BGR for OpenCV)
COLOR_OCCUPIED = (0, 0, 220)  # red (BGR for OpenCV)


# ---------------------------------------------------------------------------
# Model + slot loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classification_model(model_path):
    """Load the trained occupancy classifier. Returns None if unavailable."""
    if TF_AVAILABLE and os.path.exists(model_path):
        return load_model(model_path)
    return None


@st.cache_data
def load_slot_definitions(slots_path):
    """Load slot bounding boxes from slots.json. Returns None if unavailable."""
    if os.path.exists(slots_path):
        with open(slots_path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------
def detect_slots_classification(image_bgr, slots, model):
    """
    Crop-based classification approach.
    For each slot bounding box: crop -> preprocess -> classify -> label.

    Returns a list of dicts: {"id", "x", "y", "w", "h", "status", "confidence"}
    """
    results = []
    for slot in slots:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        crop = image_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, IMG_SIZE)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        if model is not None:
            arr = img_to_array(crop_rgb) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr, verbose=0)[0]
            # Assumes a single sigmoid output: 0 = empty, 1 = occupied.
            # If your model uses 2-class softmax, adapt this line accordingly.
            occupied_prob = float(pred[0]) if pred.shape[0] == 1 else float(pred[1])
            status = "occupied" if occupied_prob >= 0.5 else "empty"
            confidence = occupied_prob if status == "occupied" else 1 - occupied_prob
        else:
            # DEMO FALLBACK (no trained model found): a simple brightness
            # heuristic so the app is still runnable end-to-end for testing.
            # Replace this branch entirely once your model is trained.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            status = "occupied" if gray.std() > 35 else "empty"
            confidence = 0.5

        results.append({
            "id": slot.get("id"),
            "x": x, "y": y, "w": w, "h": h,
            "status": status,
            "confidence": round(confidence, 2),
        })
    return results


def detect_slots_yolo(image_bgr, yolo_model):
    """
    Alternative: full-image object detection approach.
    Use this instead of detect_slots_classification() if you trained YOLO
    to detect slots directly (e.g. using the ultralytics package).

    Example (uncomment and adapt once you have a trained .pt weights file):

        from ultralytics import YOLO
        yolo_model = YOLO("best.pt")
        preds = yolo_model(image_bgr)[0]
        results = []
        for i, box in enumerate(preds.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            status = "occupied" if cls == 1 else "empty"
            results.append({
                "id": i, "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "status": status, "confidence": round(float(box.conf[0]), 2),
            })
        return results
    """
    raise NotImplementedError("Plug in your trained YOLO model here.")


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def draw_overlays(image_bgr, results):
    """Draw colour-coded bounding boxes: green = empty, red = occupied."""
    annotated = image_bgr.copy()
    for r in results:
        color = COLOR_EMPTY if r["status"] == "empty" else COLOR_OCCUPIED
        top_left = (r["x"], r["y"])
        bottom_right = (r["x"] + r["w"], r["y"] + r["h"])
        cv2.rectangle(annotated, top_left, bottom_right, color, 2)
    return annotated


# ---------------------------------------------------------------------------
# Analytics: counts, utilisation %, congestion level, recommendation
# ---------------------------------------------------------------------------
def compute_utilization(results):
    total = len(results)
    occupied = sum(1 for r in results if r["status"] == "occupied")
    available = total - occupied
    occupancy_pct = round((occupied / total) * 100, 1) if total > 0 else 0.0
    return total, occupied, available, occupancy_pct


def classify_congestion(occupancy_pct):
    if occupancy_pct <= LOW_THRESHOLD:
        return "Low"
    elif occupancy_pct <= HIGH_THRESHOLD:
        return "Moderate"
    else:
        return "High"


def generate_recommendation(congestion_level, available):
    if congestion_level == "High":
        return "🔴 Parking full — try another area."
    elif congestion_level == "Moderate":
        return f"🟠 Limited space — {available} slot(s) left, park soon."
    else:
        return f"🟢 Slots available — proceed to park ({available} free)."


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")

    st.title("🅿️ ParkVision AI — Smart Parking Analytics")
    st.caption(
        "Upload a parking lot image to detect slot-level occupancy, "
        "view live analytics, and get a parking recommendation."
    )

    model = load_classification_model(MODEL_PATH)
    slots = load_slot_definitions(SLOTS_PATH)

    if model is None:
        st.warning(
            f"⚠️ No trained model found at '{MODEL_PATH}'. Running in DEMO mode "
            "with a simple brightness-based fallback. Add your trained model "
            "file to enable real predictions."
        )
    if slots is None:
        st.error(
            f"❌ No slot definitions found at '{SLOTS_PATH}'. Please add a "
            "slots.json file describing each parking slot's bounding box."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a parking lot image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing parking slots..."):
            results = detect_slots_classification(image_bgr, slots, model)
            annotated_bgr = draw_overlays(image_bgr, results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            total, occupied, available, occupancy_pct = compute_utilization(results)
            congestion = classify_congestion(occupancy_pct)
            recommendation = generate_recommendation(congestion, available)

        # --- Image display: original vs annotated side by side ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image_pil, use_container_width=True)
        with col2:
            st.subheader("Detected Slots")
            st.image(annotated_rgb, use_container_width=True)

        st.divider()

        # --- Key metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Slots", total)
        m2.metric("Occupied", occupied)
        m3.metric("Available", available)
        m4.metric("Occupancy", f"{occupancy_pct}%")

        # --- Congestion + recommendation ---
        st.divider()
        congestion_colors = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}
        st.subheader(f"{congestion_colors[congestion]} Congestion Level: {congestion}")
        st.progress(min(int(occupancy_pct), 100))
        st.info(recommendation)

    else:
        st.info("👆 Upload a parking lot image to get started.")


if __name__ == "__main__":
    main()
