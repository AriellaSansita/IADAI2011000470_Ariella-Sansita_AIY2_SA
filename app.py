"""
ParkVision AI - Intelligent Urban Parking Analytics & Space Optimisation
--------------------------------------------------------------------------
Pure Python / Streamlit app. Slots are detected automatically by a YOLO
model trained on the PKLot dataset (Step 3) - no manual slot-coordinate
file needed. The model detects each parking slot's location AND whether
it's occupied or empty, in one pass over the full image.

Pipeline:
  1. Upload a parking lot image
  2. YOLO detects every slot + its status (occupied / empty)
  3. Draw colour-coded boxes (green = empty, red = occupied)
  4. Compute total / occupied / available slots + utilisation %
  5. Classify congestion level (Low / Moderate / High)
  6. Generate a recommendation
  7. Display everything in a Streamlit dashboard

HOW TO PLUG IN YOUR TRAINED MODEL
-----------------------------------
Train a YOLO model (e.g. with the ultralytics package) on the PKLot
dataset with two classes: "space-empty" and "space-occupied". Export the
weights as "parking_yolo.pt" and place it next to this script. That's it -
no other config files are required.

Run with:
    streamlit run app.py
"""

import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "parking_yolo.pt"   # your trained YOLO weights (Step 3)
CONF_THRESHOLD = 0.25             # minimum detection confidence to keep a box
IMG_SIZE = 1600                   # inference resolution (higher = catches smaller/farther cars)

LOW_THRESHOLD = 40       # % occupancy below this  -> Low congestion
HIGH_THRESHOLD = 75      # % occupancy above this   -> High congestion

COLOR_EMPTY = (0, 200, 0)      # green (BGR for OpenCV)
COLOR_OCCUPIED = (0, 0, 220)   # red (BGR for OpenCV)

# Class-name -> status mapping. These match the categories in your
# _annotations.coco.json (space-empty / space-occupied).
CLASS_TO_STATUS = {
    "space-empty": "empty",
    "space-occupied": "occupied",
}


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path):
    """Load the trained YOLO model. Returns None if unavailable."""
    if YOLO_AVAILABLE and os.path.exists(model_path):
        return YOLO(model_path)
    return None


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------
def detect_slots(image_bgr, model):
    """
    Run YOLO on the full image and return a list of detected slots:
    [{"x", "y", "w", "h", "status", "confidence"}, ...]

    No slot coordinates need to be provided - YOLO finds them.
    """
    if model is not None:
        preds = model(image_bgr, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
        results = []
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()
            status = CLASS_TO_STATUS.get(class_name, class_name)
            confidence = float(box.conf[0])

            results.append({
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "status": status,
                "confidence": round(confidence, 2),
            })
        return results

    # DEMO FALLBACK (no trained model found yet): generates a handful of
    # pseudo-random slots so the rest of the app is testable end-to-end.
    # Delete this branch once your YOLO weights are trained and in place.
    h_img, w_img = image_bgr.shape[:2]
    rng = np.random.default_rng(seed=42)
    demo_results = []
    grid_cols, grid_rows = 4, 2
    cell_w, cell_h = w_img // grid_cols, h_img // grid_rows
    for r in range(grid_rows):
        for c in range(grid_cols):
            status = "occupied" if rng.random() > 0.5 else "empty"
            demo_results.append({
                "x": c * cell_w + 5, "y": r * cell_h + 5,
                "w": cell_w - 10, "h": cell_h - 10,
                "status": status, "confidence": 0.5,
            })
    return demo_results


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
        "Upload a parking lot image. YOLO detects every slot and its "
        "occupancy status automatically — no manual setup required."
    )

    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"⚠️ No trained YOLO weights found at '{MODEL_PATH}'. Running in "
            "DEMO mode with randomly generated slots so you can test the "
            "app. Add your trained 'parking_yolo.pt' to enable real detection."
        )

    uploaded_file = st.file_uploader(
        "Upload a parking lot image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting parking slots..."):
            results = detect_slots(image_bgr, model)
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
