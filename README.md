# Urban Object Detection — YOLO11 Inspection Panel

A Streamlit demo for a **YOLO11 model fine-tuned on 26 classes of urban objects** — street
furniture, vehicles, traffic signals, and pedestrian hazards. Upload a photo or pick one of the
bundled samples, move the confidence threshold, and the app draws every detection with its class
label and confidence score, plus a per-class count.

**▶ [Live demo — urban-object-detection.streamlit.app](https://urban-object-detection.streamlit.app/)**

[![Live app](https://img.shields.io/badge/Live_demo-urban--object--detection.streamlit.app-F5B942?logo=streamlit&logoColor=white)](https://urban-object-detection.streamlit.app/)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-0E1113)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-F5B942)
![torch](https://img.shields.io/badge/torch-2.8.0%2Bcpu-0E1113)
![mAP](https://img.shields.io/badge/test_mAP@0.5:0.95-0.510-0E1113)

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

## Results

Full numbers are in [`metrics/`](metrics/) — `metrics_summary.csv` and `per_class_ap.csv`. The app
reads its headline figure straight from that CSV rather than hardcoding it, so the number on the
page and the number in this table cannot drift apart.

| Metric | Validation | Test |
|---|---:|---:|
| **mAP@0.5:0.95** (primary) | 0.4661 | **0.5102** |
| mAP@0.5 | 0.6503 | 0.6964 |
| mAP@0.75 | 0.5016 | 0.5576 |
| Precision @conf=0.25 | 0.7803 | 0.8160 |
| Recall @conf=0.25 | 0.6502 | 0.7156 |
| Inference (ms/image) | 11.41 | 12.78 |

Two caveats worth stating before the per-class breakdown:

- **The headline mAP covers 24 of the 26 classes, not all 26.** `Bench` and `Umbrella` have zero
  instances in the test split, so they are absent from the evaluation entirely. The macro mean of
  the 24 evaluated classes reproduces the reported 0.5102 exactly. Counting the two absent classes
  as zero would give 0.4709 instead — neither number is wrong, but they answer different questions,
  and the model's behaviour on those two classes is simply unmeasured here.
- **Test scores exceed validation on every metric.** That is the opposite of the usual direction
  and suggests the test split is the easier of the two rather than that the model generalises
  unusually well. Treat the validation column as the more conservative estimate.

### Per-class AP (test split, 3,701 boxes across 24 classes)

| Class | AP@0.5:0.95 | AP@0.5 | Precision | Recall | Boxes |
|---|---:|---:|---:|---:|---:|
| Tree | 0.878 | 0.955 | 0.942 | 0.960 | 101 |
| Stop Sign | 0.824 | 0.984 | 0.973 | 0.986 | 74 |
| Crosswalk | 0.814 | 0.962 | 0.979 | 0.952 | 186 |
| Door | 0.720 | 0.893 | 0.945 | 0.892 | 120 |
| Rat | 0.700 | 0.880 | 0.905 | 0.885 | 129 |
| Fire Hydrant | 0.662 | 0.982 | 0.897 | 1.000 | 103 |
| Red Light | 0.650 | 0.913 | 0.953 | 0.904 | 134 |
| Elevator | 0.649 | 0.776 | 0.911 | 0.782 | 197 |
| Traffic Cone | 0.596 | 0.830 | 0.926 | 0.835 | 231 |
| Yellow Light | 0.555 | 0.727 | 0.866 | 0.736 | 106 |
| Bus | 0.530 | 0.631 | 0.693 | 0.676 | 139 |
| Car | 0.490 | 0.651 | 0.706 | 0.708 | 305 |
| Green Light | 0.490 | 0.706 | 0.736 | 0.744 | 154 |
| Gun | 0.484 | 0.663 | 0.905 | 0.667 | 129 |
| Bushes | 0.463 | 0.637 | 0.864 | 0.652 | 221 |
| Scooter | 0.415 | 0.635 | 0.865 | 0.612 | 147 |
| Pothole | 0.408 | 0.623 | 0.697 | 0.661 | 109 |
| Train | 0.391 | 0.718 | 0.841 | 0.738 | 122 |
| Bicycle | 0.373 | 0.618 | 0.725 | 0.624 | 194 |
| Motorcycle | 0.315 | 0.660 | 0.727 | 0.676 | 250 |
| Truck | 0.294 | 0.352 | 0.917 | 0.360 | 50 |
| Branch | 0.210 | 0.365 | 0.624 | 0.399 | 54 |
| Stairs | 0.203 | 0.314 | 0.467 | 0.440 | 25 |
| Person | 0.131 | 0.240 | 0.520 | 0.285 | 421 |

### What the per-class numbers actually say

**`Person` is the model's worst class by a wide margin — and it has the most test instances.**
AP 0.131 on 421 boxes, with recall at 0.285. More data did not help here, which points at the
label distribution rather than sample count: people appear in these street scenes at every scale,
heavily occluded and overlapping, and at 300 px many are only a handful of pixels tall. This is the
single clearest target for improvement, and it is worth knowing before anyone points the demo at a
crowded pavement.

**Some classes are conservative rather than bad.** `Truck` has precision 0.917 against recall 0.360
— when it fires it is almost always right, but it misses two-thirds of trucks, most likely
surrendering them to `Car` and `Bus`. `Gun` (0.905 / 0.667) and `Scooter` (0.865 / 0.612) show the
same shape. For a hazard-detection framing, that trade-off is the wrong way round: you would rather
over-flag than miss, which argues for a lower threshold than the 0.25 default.

**A high AP@0.5 with a low AP@0.5:0.95 means loose boxes.** `Fire Hydrant` finds every instance
(recall 1.000, AP@0.5 0.982) but drops to 0.662 under the stricter IoU sweep. `Motorcycle` loses
0.345 between the two, `Train` 0.327. These objects are being *located* reliably and *outlined*
imprecisely — a different failure from missing them, and one that matters far less for a
"what's in this image" demo than it would for measurement or tracking.

**The top of the table is dominated by large, rigid, high-contrast objects** — `Tree`, `Stop Sign`,
`Crosswalk`, `Door` all clear AP 0.72. That is the expected shape for a model trained at 300 px:
it does well on things that survive aggressive downscaling.

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
risks tripping the free tier's storage limit.

### Why `packages.txt` exists

```
libgl1
libglib2.0-0
```

`ultralytics` declares `opencv-python>=4.6.0` as a hard dependency, so the regular (non-headless)
OpenCV build is always installed — listing `opencv-python-headless` in `requirements.txt` does
*not* displace it. Both distributions write into the same `cv2/` directory, and whichever lands
last wins, so the headless swap that works on some deploys is really just resolver luck.

Regular OpenCV needs `libGL.so.1`, which the Streamlit Cloud container does not ship. `packages.txt`
is installed with apt before the Python dependencies and supplies it, which fixes the import
deterministically instead of hoping the right OpenCV wins. `opencv-python` is also version-pinned,
because left unpinned the resolver picks up OpenCV 5.x — released after `ultralytics 8.4.67` and
not validated against it.

## Project structure

```
app.py                   # the whole app: UI, inference, and the overlay renderer
requirements.txt         # pinned dependencies, CPU-only torch
packages.txt             # apt packages for Streamlit Cloud (libGL for OpenCV)
README.md
.streamlit/config.toml   # theme colours — keep in sync with the palette in app.py
models/best.pt           # fine-tuned YOLO11 checkpoint (~19 MB)
metrics/                 # evaluation results; app.py reads its headline mAP from here
sample_images/           # 5 test images for the "choose a sample" mode
data.yaml                # 26-class reference — documentation only, not read at runtime
```

## Sample images

`sample_images/` holds five test images taken from the dataset. Two of them
(`sample_02_traffic_cone.jpg`, `sample_03_senjata.jpg`) still carry the stock-photo watermark that
came with the source dataset — swap them for your own photographs if licensing matters for your
use of this repo.
