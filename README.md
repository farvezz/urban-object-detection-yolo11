# Urban Object Detection — YOLO11 Inspection Panel

A Streamlit demo for a **YOLO11 model fine-tuned on 26 classes of urban objects** — street
furniture, vehicles, traffic signals, and pedestrian hazards. Upload a photo or pick one of the
bundled samples, move the confidence threshold, and the app draws every detection with its class
label and confidence score, plus a per-class count.

![Streamlit](https://img.shields.io/badge/Streamlit-1.58-0E1113)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-F5B942)
![torch](https://img.shields.io/badge/torch-2.8.0%2Bcpu-0E1113)

> **Note on language:** the app's interface is in **Indonesian**. This README is the English
> project write-up.

---

## What this project is

Object detection models are usually demonstrated through a notebook: a few test images, a grid of
outputs, a confusion matrix. That shows the model works, but it doesn't let anyone *interrogate* it.

This app closes that gap. It takes the trained checkpoint and puts it behind an interface where a
non-technical viewer can push their own images through the model and immediately see two things:
what it detects, and where its limits are. The threshold slider matters here — moving it from 0.25
to 0.60 makes the precision/recall trade-off tangible in a way a static metric doesn't.

The design deliberately foregrounds uncertainty rather than hiding it. Every detection carries its
raw confidence score, the class list is stated up front, and the interface says plainly that
objects outside those 26 classes will either be missed or misclassified as a visually similar
class. A demo that only shows successful detections is a worse portfolio piece than one that shows
you understand the failure modes.

## The model

| | |
|---|---|
| **Architecture** | YOLO11, fine-tuned |
| **Classes** | 26 |
| **Dataset** | Roboflow export; COCO-format annotations converted to YOLO format |
| **Training resolution** | 300 × 300 px (the dataset's native size) |
| **Checkpoint** | `models/best.pt`, ~19 MB, committed directly (no Git LFS needed) |
| **Inference device** | CPU — no GPU required |

### The 26 classes

Index order matters: it has to match the trained checkpoint. The app reads this list from
`model.names` at runtime rather than hardcoding it, so it can never drift from the weights that
are actually loaded.

```
00 Bench          07 Door           14 Pothole        21 Train
01 Bicycle        08 Elevator       15 Rat            22 Tree
02 Branch         09 Fire Hydrant   16 Red Light      23 Truck
03 Bus            10 Green Light    17 Scooter        24 Umbrella
04 Bushes         11 Gun            18 Stairs         25 Yellow Light
05 Car            12 Motorcycle     19 Stop Sign
06 Crosswalk      13 Person         20 Traffic Cone
```

### Limitations

These are stated in the app itself, not buried here:

- **Closed vocabulary.** Anything outside the 26 classes above generally won't be detected. When it
  *is* detected, it is usually misclassified into the nearest visually similar class — a van reads
  as `Truck`, a lamp post as `Branch`.
- **Scale mismatch.** Every training image was 300 × 300 px. On a high-resolution photo, small or
  distant objects — traffic lights, traffic cones, fire hydrants — occupy far fewer relative pixels
  than anything the model saw during training, and are frequently missed. Lowering the threshold
  recovers some of them at the cost of more false positives.
- **Confidence is not correctness.** The score is the model's raw output. A confident wrong answer
  is entirely possible, which is exactly why the app shows the number instead of a verdict.

No aggregate evaluation metrics (mAP, per-class precision/recall) are published in this repo — the
demo is about qualitative behaviour, and quoting numbers without the eval split alongside them
would be misleading.

---

## Features

| | |
|---|---|
| **Two input modes** | Upload `jpg` / `jpeg` / `png`, or choose from 5 bundled sample images |
| **EXIF orientation correction** | Phone photos stored sideways are uprighted via `ImageOps.exif_transpose` |
| **Confidence threshold** | Slider from 0.05 to 0.95 (default 0.25); results update instantly, no "Run" button |
| **Custom detection overlay** | Corner-bracket boxes with a class label and confidence score, drawn with Pillow |
| **Per-class counts** | Lightweight HTML/CSS bar rows, sorted by count descending |
| **Distinct empty states** | Separate guidance for "no image yet" and "no detections above threshold" |
| **Download result** | Save the annotated image as a PNG |

## How it works

```
image in
  → ImageOps.exif_transpose        fix orientation
  → model.predict(conf=0.05)       one inference per image, cached
  → filter in Python by slider     moving the slider does not re-run the model
  → draw_overlay()                 corner brackets + labels, drawn with Pillow
  → per-class counts
```

Two decisions shape the code.

**Inference runs once per image, not once per slider move.** The model is called at the slider's
floor (`CONF_FLOOR = 0.05`) and the resulting detection list is cached with `@st.cache_data`, keyed
on a digest of the image content. Moving the slider just filters that cached list. This is
*equivalent* to re-running the model at a higher threshold — NMS always processes the
highest-confidence boxes first, so a box that survives at a high threshold never depends on the
lower-confidence boxes below it — but the slider responds instantly instead of triggering a fresh
forward pass. The checkpoint itself loads through `@st.cache_resource`, so it is read from disk
once per process.

**Detections are drawn by hand, not by `result.plot()`.** `draw_overlay()` first rescales the image
to a 900–1400 px canvas. This matters more than it sounds: the training images are 300 px, so
annotating at native size and letting the browser upscale produces soft, blocky boxes and text. It
then draws a thin outline around the full box, thick brackets at the four corners, and a single
label tab holding the class name (sans-serif) and the confidence score (monospace). Larger objects
are drawn first so a small object's label is never buried under a bigger box, and a label that
would clip off the top edge is moved inside the box instead.

## Design notes

A dark neutral palette with exactly one accent colour — roadwork amber `#F5B942` — applied
consistently across both the interface and the detection overlay. Hairline rules instead of drop
shadowed cards. Monospace type (JetBrains Mono) is reserved strictly for numeric data: confidence
scores, object counts, inference time, class indices. Everything else is sans-serif (Inter). The
aesthetic is meant to read as a technical inspection panel, which suits a dataset about street
hazards and infrastructure.

The palette constants at the top of `app.py` and the colours in `.streamlit/config.toml` must stay
in sync, otherwise Streamlit's built-in widgets stop matching the custom CSS. Streamlit's default
chrome (hamburger menu, toolbar, "made with Streamlit" footer) is hidden.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. There is no build step and no training at runtime — the
`models/best.pt` checkpoint is loaded directly.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository. `models/best.pt` (~19 MB) is comfortably under GitHub's
   100 MB file limit, so commit it directly — **no Git LFS**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick this repo, and set
   **Main file path** to `app.py`.
3. Under **Advanced settings**, set the Python version to **3.12**. The pinned wheels are the ones
   verified against 3.12; leaving it on a different interpreter risks `torch==2.8.0+cpu` failing to
   resolve at build time.
4. The first build takes a few minutes, since it installs torch and ultralytics.

### Why the `--extra-index-url` line in `requirements.txt` must stay

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0+cpu
torchvision==0.23.0+cpu
```

Without it, pip resolves the CUDA build of torch and drags in roughly 2 GB of NVIDIA libraries.
Streamlit Community Cloud has no GPU, so none of it is used — it only slows the build down and
risks tripping the free tier's storage limit. `opencv-python-headless` is pinned instead of
`opencv-python` because the Streamlit Cloud container has no `libGL`.

## Project structure

```
app.py                   # the whole app: UI, inference, and the overlay renderer
requirements.txt         # pinned dependencies, CPU-only torch
README.md
.streamlit/config.toml   # theme colours — keep in sync with the palette in app.py
models/best.pt           # fine-tuned YOLO11 checkpoint (~19 MB)
sample_images/           # 5 test images for the "choose a sample" mode
data.yaml                # 26-class reference — documentation only, not read at runtime
```

## Sample images

`sample_images/` holds five test images taken from the dataset. Two of them
(`sample_02_traffic_cone.jpg`, `sample_03_senjata.jpg`) still carry the stock-photo watermark that
came with the source dataset — swap them for your own photographs if licensing matters for your
use of this repo.
