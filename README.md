# 🅿️ ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

**Live App:** [PASTE YOUR STREAMLIT CLOUD LINK HERE]
**GitHub Repository:** [PASTE YOUR GITHUB REPO LINK HERE]

---

## 📋 Student Details

- **Full Name:** [YOUR FULL NAME]
- **Candidate Registration Number:** [YOUR REGISTRATION NUMBER]
- **CRS Name:** Artificial Intelligence
- **Course Name:** [YOUR COURSE NAME]
- **School Name:** [YOUR SCHOOL NAME]

---

## 📖 Project Overview

ParkVision AI is a computer vision system built for UrbanFlow AI's smart-city
initiative, designed to solve a common urban problem: drivers and city
planners have no real-time visibility into how full a parking lot actually
is, slot by slot. Instead of showing only a single "lot full / lot open"
status, this project detects the occupancy of **individual parking slots**
from an aerial camera image, then turns those raw detections into useful,
human-readable insights — occupancy percentage, congestion level, and a
plain-language parking recommendation.

The system takes a parking lot image as input and returns:

- The location and status (empty / occupied) of every detected slot
- A colour-coded annotated image (green = empty, red = occupied)
- Total / occupied / available slot counts
- Occupancy percentage and a Low / Moderate / High congestion rating
- A recommendation (e.g. "Slots available — proceed" or "Parking full — try another area")

---

## 🎯 Problem Statement

Traditional parking systems only report whether a lot is generally full or
not, which forces drivers to search blindly and leaves lot operators without
granular data to manage congestion. ParkVision AI addresses this by
performing **slot-level** occupancy detection directly from a single
overhead image, so that both individual users and city-level dashboards can
get an accurate, real-time picture of available space — without needing
in-ground sensors at every slot.

---

## 🔍 Research Findings & References

- The project uses **object detection (YOLO)** rather than a per-slot
  classification approach. A classification model can only label a
  pre-cropped image as empty/occupied, which requires manually defining
  every slot's coordinates for each camera view. A detection model instead
  locates and classifies every slot directly from the full image, which
  generalises better to a live camera feed and needs no manual slot
  configuration.
- **Dataset:** [PKLot Dataset on Kaggle](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset)
  — a widely used benchmark dataset for parking-slot occupancy detection,
  originally introduced in Almeida et al., *"PKLot – A robust dataset for
  parking lot classification,"* Expert Systems with Applications, 2015.
- **Model framework:** [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
  was used as the primary reference for training configuration, data format
  requirements, and inference parameters (confidence threshold, image size).
- **Annotation format:** the dataset was provided as a Roboflow export in
  **COCO format** (`_annotations.coco.json`), which required a custom
  conversion step to YOLO's normalized bounding-box `.txt` label format
  before training (see *Data Preparation* below).

---

## 🗂️ Data Preparation

1. **Source:** the PKLot dataset was downloaded via the Kaggle API directly
   into Google Colab. The available data was provided as a single
   `train.zip` containing:
   - **8,691 images**, all from a fixed overhead security-camera view of a
     parking lot
   - **497,856 bounding-box annotations** in COCO format
   - Two real classes: `space-empty` and `space-occupied` (plus a parent
     `spaces` category that was excluded from training)

2. **COCO → YOLO conversion:** since YOLO training requires normalized
   `(class, x_center, y_center, width, height)` label files rather than
   COCO's `[x, y, width, height]` pixel-based format, a conversion script
   was written to:
   - Map `space-empty` → class `0` and `space-occupied` → class `1`
   - Convert every bounding box to YOLO's normalized format per image
   - Write one `.txt` label file per image, matching YOLO's expected folder
     structure (`images/`, `labels/`)

3. **Validation split:** because only a training set was provided (no
   separate validation/test set), **15% of the images were randomly held
   out** to create a validation split, so model performance could be
   measured on unseen data during training.

4. **Config file:** a `data.yaml` file was generated to point YOLO at the
   correct image folders and define the two class names.

---

## 🧠 Model & Training Details

| Parameter | Value |
|---|---|
| Model architecture | YOLOv8n (nano) |
| Framework | Ultralytics YOLO |
| Training platform | Google Colab (free-tier GPU, T4) |
| Image size | 640×640 (training) |
| Epochs | 30 |
| Batch size | 16 |
| Classes | `space-empty`, `space-occupied` |

**Why YOLOv8n?** The nano variant was chosen for fast training on Colab's
free GPU tier and quick iteration, given the project's timeline. A larger
variant (e.g. YOLOv8s or YOLOv8m) would likely improve detection accuracy,
particularly for small/distant slots, at the cost of longer training time —
this is noted as a direction for future improvement.

---

## 📊 Performance Metrics

[FILL IN: paste your final training metrics here, e.g. from the last epoch
of training or from Ultralytics' auto-generated results — mAP50, mAP50-95,
precision, recall. You can find these printed at the end of your Cell 4
training output, or in the results.png / results.csv file YOLO saves to
runs/detect/parkvision_yolo/]

**Qualitative testing summary:**
- Performs reliably on images from the same camera framing as the training
  data, across varying occupancy levels and lighting conditions.
- Detection coverage decreases for very small/distant slots in
  high-density, high-resolution images at the default inference size —
  mitigated in the app by an adjustable inference resolution setting.
- Does not generalise to parking lots captured from a substantially
  different camera angle (e.g. oblique/side-angle CCTV) that the model was
  never trained on — a known and expected limitation of the dataset's
  fixed-camera nature (see *Limitations* below).

---

## ✨ App Features

- **Batch upload** — process multiple images in one session, with a
  per-image breakdown and a combined summary table
- **Adjustable confidence threshold** — tune how strict detection is,
  directly from the sidebar, without redeploying
- **Adjustable inference resolution** — trade off speed vs. detection
  accuracy on smaller/distant slots, directly from the sidebar
- **Confidence-based colour gradient** — box colour intensity reflects the
  model's confidence, so uncertain detections are visually distinct from
  confident ones
- **Occupancy history chart** — tracks occupancy percentage across every
  image processed in a session
- **Automatic congestion classification** — Low / Moderate / High, based on
  configurable occupancy thresholds
- **Plain-language recommendation** — a simple, actionable suggestion
  generated from the congestion level

---

## 🖼️ Screenshots

- Upload screen
- Detected slots (annotated image with colour-coded boxes)
- Metrics, congestion level, and recommendation section
- Sidebar settings panel

---

## ⚠️ Known Limitations

- **Fixed-camera dependency:** the model performs best within the exact
  framing of its training camera. Regions at the very edge of a wider or
  differently-cropped image (outside what the training images ever showed)
  are not reliably detected, since the model has no examples of that
  content.
- **Camera-angle sensitivity:** the model was trained exclusively on
  overhead/bird's-eye imagery and does not generalise to oblique or
  angled camera views, which look visually very different.
- **Hosting constraints:** the app runs on Streamlit Community Cloud's free
  tier, which is CPU-only and has resource limits — high-resolution
  inference is noticeably slower than it would be on a GPU-backed server.

These are documented as intentional scope boundaries rather than unresolved
bugs — real-world deployments of systems like this are similarly typically
calibrated to a specific, fixed camera per site.

---

## 🚀 How to Run Locally

```bash
git clone [YOUR REPO URL]
cd [YOUR REPO FOLDER]
pip install -r requirements.txt
streamlit run app.py
```

**Note:** the trained model file (`parking_yolo.pt`) must be present in the
same folder as `app.py` for real detections to run. Without it, the app
falls back to a demo mode with randomly generated slots so the interface
can still be tested.

On Streamlit Community Cloud specifically, `packages.txt` (listing system
libraries like `libgl1`) is also required alongside `requirements.txt` for
OpenCV to import correctly in the hosted environment.
