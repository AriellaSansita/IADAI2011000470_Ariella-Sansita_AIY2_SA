"""
ParkVision AI - Intelligent Urban Parking Analytics & Space Optimisation
--------------------------------------------------------------------------
Pure Python / Streamlit app. Slots are detected automatically by a YOLO
model trained on the PKLot dataset (Step 3) - no manual slot-coordinate
file needed.

Features:
  - Batch upload: process multiple images at once, with a summary table
  - Confidence-based color gradient: box color intensity reflects confidence
  - Occupancy history chart: tracks occupancy % across images in this session
  - Adjustable confidence threshold and inference resolution (sidebar)

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
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
MODEL_PATH = "parking_yolo.pt"

DEFAULT_CONF = 0.25
DEFAULT_IMG_SIZE = 1280

LOW_THRESHOLD = 40
HIGH_THRESHOLD = 75

# Base hues for the confidence gradient (BGR for OpenCV)
EMPTY_HUE = (0, 255, 0)       # pure green at max confidence
OCCUPIED_HUE = (0, 0, 255)    # pure red at max confidence
LOW_CONF_GREY = (140, 140, 140)  # low-confidence boxes fade toward grey

CLASS_TO_STATUS = {
    "space-empty": "empty",
    "space-occupied": "occupied",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path):
    if YOLO_AVAILABLE and os.path.exists(model_path):
        return YOLO(model_path)
    return None


# ---------------------------------------------------------------------------
# Slot detection
# ---------------------------------------------------------------------------
def detect_slots(image_bgr, model, conf_threshold, img_size):
    if model is not None:
        preds = model(image_bgr, conf=conf_threshold, imgsz=img_size, verbose=False)[0]
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

    # DEMO FALLBACK (no trained model found yet)
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
                "status": status, "confidence": round(rng.uniform(0.4, 0.95), 2),
            })
    return demo_results


# ---------------------------------------------------------------------------
# Overlay drawing with confidence-based color gradient
# ---------------------------------------------------------------------------
def _blend_color(base_hue, confidence):
    """Interpolate between low-confidence grey and the full-strength hue."""
    t = max(0.0, min(1.0, confidence))  # clamp 0-1
    return tuple(
        int(LOW_CONF_GREY[i] + (base_hue[i] - LOW_CONF_GREY[i]) * t)
        for i in range(3)
    )


def draw_overlays(image_bgr, results, show_confidence):
    annotated = image_bgr.copy()
    for r in results:
        base_hue = EMPTY_HUE if r["status"] == "empty" else OCCUPIED_HUE
        color = _blend_color(base_hue, r["confidence"])
        thickness = 3 if r["confidence"] >= 0.6 else 1  # faint boxes also drawn thinner

        top_left = (r["x"], r["y"])
        bottom_right = (r["x"] + r["w"], r["y"] + r["h"])
        cv2.rectangle(annotated, top_left, bottom_right, color, thickness)

        if show_confidence:
            label = f"{r['confidence']:.2f}"
            cv2.putText(annotated, label, (r["x"], max(r["y"] - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return annotated


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


def process_image(image_pil, model, conf_threshold, img_size, show_confidence):
    """Run the full pipeline on a single PIL image and return everything the UI needs."""
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    results = detect_slots(image_bgr, model, conf_threshold, img_size)
    annotated_bgr = draw_overlays(image_bgr, results, show_confidence)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    total, occupied, available, occupancy_pct = compute_utilization(results)
    congestion = classify_congestion(occupancy_pct)
    recommendation = generate_recommendation(congestion, available)
    return {
        "annotated_rgb": annotated_rgb,
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy_pct": occupancy_pct,
        "congestion": congestion,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")

    # Session-level history for the occupancy trend chart
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"label", "occupancy_pct", "timestamp"}

    st.title("🅿️ ParkVision AI — Smart Parking Analytics")
    st.caption(
        "Upload one or more parking lot images. YOLO detects every slot and "
        "its occupancy status automatically — no manual setup required."
    )

    # --- Sidebar: grouped, labeled controls ---
    with st.sidebar:
        st.header("⚙️ Detection Settings")

        with st.expander("🎯 Sensitivity", expanded=True):
            conf_threshold = st.slider(
                "Confidence threshold", 0.05, 0.90, DEFAULT_CONF, 0.05,
                help="Lower catches more slots but may add noisy detections. "
                     "Higher keeps only confident detections."
            )

        with st.expander("🔍 Detail level", expanded=True):
            img_size = st.select_slider(
                "Inference resolution",
                options=[640, 960, 1280, 1600, 1920],
                value=DEFAULT_IMG_SIZE,
                help="Higher resolution catches smaller/farther slots in wide "
                     "aerial shots, but runs slower."
            )

        with st.expander("🎨 Display"):
            show_confidence = st.checkbox("Show confidence scores on boxes", value=False)
            st.caption("Box color intensity already reflects confidence — "
                       "faint/grey boxes are low-confidence detections.")

        st.divider()
        if st.session_state.history:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.rerun()

        st.caption(f"Model: {MODEL_PATH}")
        st.caption(f"Congestion bands: Low ≤{LOW_THRESHOLD}% · High >{HIGH_THRESHOLD}%")

    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.warning(
            f"⚠️ No trained YOLO weights found at '{MODEL_PATH}'. Running in "
            "DEMO mode with randomly generated slots so you can test the app."
        )

    uploaded_files = st.file_uploader(
        "Upload parking lot image(s)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        batch_rows = []

        for uploaded_file in uploaded_files:
            image_pil = Image.open(uploaded_file).convert("RGB")

            with st.spinner(f"Detecting slots in {uploaded_file.name}..."):
                result = process_image(image_pil, model, conf_threshold, img_size, show_confidence)

            # Log to session history for the trend chart
            st.session_state.history.append({
                "label": uploaded_file.name,
                "occupancy_pct": result["occupancy_pct"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

            batch_rows.append({
                "Image": uploaded_file.name,
                "Total": result["total"],
                "Occupied": result["occupied"],
                "Available": result["available"],
                "Occupancy %": result["occupancy_pct"],
                "Congestion": result["congestion"],
            })

            with st.expander(f"📷 {uploaded_file.name} — {result['occupancy_pct']}% occupied", expanded=(len(uploaded_files) == 1)):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_pil, caption="Uploaded", use_container_width=True)
                with col2:
                    st.image(result["annotated_rgb"], caption="Detected Slots", use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Slots", result["total"])
                m2.metric("Occupied", result["occupied"])
                m3.metric("Available", result["available"])
                m4.metric("Occupancy", f"{result['occupancy_pct']}%")

                congestion_colors = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}
                st.write(f"**{congestion_colors[result['congestion']]} Congestion Level: {result['congestion']}**")
                st.progress(min(int(result["occupancy_pct"]), 100))
                st.info(result["recommendation"])

        # --- Batch summary table (only meaningful with >1 image) ---
        if len(uploaded_files) > 1:
            st.divider()
            st.subheader("📊 Batch Summary")
            df = pd.DataFrame(batch_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # --- Occupancy history chart ---
        if len(st.session_state.history) > 1:
            st.divider()
            st.subheader("📈 Occupancy History (this session)")
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df["point"] = hist_df["label"] + " (" + hist_df["timestamp"] + ")"
            st.line_chart(hist_df.set_index("point")["occupancy_pct"])
            st.caption("Tracks occupancy % across every image processed this session, in order.")

    else:
        st.info("👆 Upload one or more parking lot images to get started.")


if __name__ == "__main__":
    main()
