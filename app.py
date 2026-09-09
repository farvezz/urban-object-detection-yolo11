"""
Panel Inspeksi Objek Perkotaan
==============================

Demo Streamlit untuk model YOLO11 yang di-fine-tune pada 26 kelas objek
perkotaan (dataset Roboflow, anotasi COCO dikonversi ke format YOLO,
citra latih 300x300 piksel).

Jalankan lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Lokasi berkas
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"
SAMPLES_DIR = APP_DIR / "sample_images"
METRICS_PATH = APP_DIR / "metrics" / "metrics_summary.csv"

SAMPLES = {
    "1. Car in the street": "sample_01_mobil.jpg",
    "2. Traffic cone row": "sample_02_traffic_cone.jpg",
    "3. Firearm": "sample_03_senjata.jpg",
    "4. Building door": "sample_04_pintu.jpg",
    "5. Green traffic light": "sample_05_lampu_hijau.jpg",
}

# Batas bawah slider. Inferensi selalu dijalankan pada nilai ini, lalu hasilnya
# disaring di Python sesuai posisi slider. NMS memproses kotak dengan confidence
# tertinggi lebih dulu, jadi menyaring belakangan memberi hasil yang sama dengan
# menjalankan ulang model pada ambang lebih tinggi -- tanpa biaya inferensi ulang
# setiap kali slider digeser.
CONF_FLOOR = 0.05
CONF_DEFAULT = 0.25

TRAIN_RES = "300 x 300"

# ---------------------------------------------------------------------------
# Palet -- satu warna aksen (kuning rambu kerja) di atas netral gelap
# ---------------------------------------------------------------------------

BG = "#0E1113"
PANEL = "#14181A"
PANEL_SOFT = "#191E20"
LINE = "#252B2E"
TEXT = "#E5EAE8"
MUTED = "#7C8A87"
ACCENT = "#F5B942"
INK = "#0C0F10"

st.set_page_config(
    page_title="Urban Object Inspection Panel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Gaya
# ---------------------------------------------------------------------------

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Sembunyikan chrome bawaan Streamlit */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; height: 0 !important; }}

html, body, .stApp, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
}}
.stApp {{ background: {BG}; color: {TEXT}; }}

.block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {{
    max-width: 1260px;
    padding: 2.4rem 1.6rem 3rem 1.6rem;
}}

/* ---- kepala halaman ---------------------------------------------------- */
.masthead {{ margin-bottom: 1.15rem; }}
.masthead .eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    color: {ACCENT};
    margin-bottom: 0.55rem;
}}
.masthead h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.022em;
    color: {TEXT};
    margin: 0;
    line-height: 1.2;
}}
.masthead p {{
    font-size: 0.9rem;
    line-height: 1.6;
    color: {MUTED};
    margin: 0.5rem 0 0 0;
    max-width: 62ch;
}}

/* ---- strip spesifikasi model ------------------------------------------- */
.specbar {{
    display: flex;
    flex-wrap: wrap;
    border-top: 1px solid {LINE};
    border-bottom: 1px solid {LINE};
    margin-bottom: 1.6rem;
}}
.spec {{
    padding: 0.6rem 1.15rem;
    border-left: 1px solid {LINE};
    flex: 0 0 auto;
}}
.spec:first-child {{ border-left: none; padding-left: 0; }}
.spec .k {{
    display: block;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    color: {MUTED};
    margin-bottom: 0.18rem;
}}
.spec .v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: {TEXT};
}}

/* ---- panel ------------------------------------------------------------- */
[data-testid="stColumn"], [data-testid="column"] {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 2px;
    padding: 1.15rem 1.25rem 1.35rem 1.25rem;
}}

.sec {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid {LINE};
    margin: 0 0 0.9rem 0;
}}
.sec .t {{ font-size: 0.86rem; font-weight: 600; color: {TEXT}; letter-spacing: -0.005em; }}
.sec .n {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {MUTED}; }}
.gap {{ height: 1.4rem; }}

/* ---- keadaan kosong ---------------------------------------------------- */
.empty {{
    border: 1px dashed {LINE};
    border-radius: 2px;
    padding: 2.2rem 1.5rem;
    text-align: center;
}}
.empty .mark {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: {ACCENT};
    margin-bottom: 0.7rem;
}}
.empty .msg {{
    font-size: 0.87rem;
    line-height: 1.65;
    color: {MUTED};
    max-width: 48ch;
    margin: 0 auto;
}}

/* ---- ringkasan angka --------------------------------------------------- */
.stats {{ display: flex; border-top: 1px solid {LINE}; margin: 1.1rem 0 1.35rem 0; }}
.stat {{ flex: 1; padding: 0.75rem 0 0.1rem 1rem; border-left: 1px solid {LINE}; }}
.stat:first-child {{ border-left: none; padding-left: 0; }}
.stat .k {{ display: block; font-size: 0.68rem; color: {MUTED}; margin-bottom: 0.3rem; }}
.stat .v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 500;
    color: {TEXT};
    line-height: 1;
}}
.stat .u {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {MUTED}; margin-left: 0.15rem; }}

/* ---- batang jumlah per kelas ------------------------------------------- */
.bar {{
    display: grid;
    grid-template-columns: 132px 1fr 30px;
    align-items: center;
    gap: 0.8rem;
    padding: 0.36rem 0;
}}
.bar + .bar {{ border-top: 1px solid {LINE}; }}
.bar .name {{ font-size: 0.82rem; color: {TEXT}; }}
.bar .track {{ background: {PANEL_SOFT}; height: 8px; }}
.bar .fill {{ background: {ACCENT}; height: 8px; display: block; }}
.bar .num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    color: {ACCENT};
    text-align: right;
}}

/* ---- daftar deteksi ---------------------------------------------------- */
.det {{
    display: grid;
    grid-template-columns: 28px 1fr auto;
    align-items: center;
    gap: 0.7rem;
    padding: 0.34rem 0;
    font-size: 0.82rem;
    color: {TEXT};
}}
.det + .det {{ border-top: 1px solid {LINE}; }}
.det .i {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {MUTED}; }}
.det .c {{ font-family: 'JetBrains Mono', monospace; font-weight: 500; color: {ACCENT}; }}

/* ---- daftar kelas ------------------------------------------------------ */
.classes {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.9rem; }}
.chip {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.4rem;
    border: 1px solid {LINE};
    background: {PANEL_SOFT};
    padding: 0.22rem 0.5rem;
    border-radius: 2px;
}}
.chip .i {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {ACCENT}; }}
.chip .n {{ font-size: 0.75rem; color: {TEXT}; }}

.note {{ font-size: 0.81rem; line-height: 1.65; color: {MUTED}; }}
.note + .note {{ margin-top: 0.6rem; }}
.note b {{ color: {TEXT}; font-weight: 500; }}

.foot {{
    border-top: 1px solid {LINE};
    margin-top: 1.9rem;
    padding-top: 0.85rem;
    font-size: 0.75rem;
    line-height: 1.6;
    color: {MUTED};
}}

/* ---- widget ------------------------------------------------------------ */
[data-testid="stFileUploaderDropzone"] {{
    background: {PANEL_SOFT};
    border: 1px dashed {LINE};
    border-radius: 2px;
}}
[data-testid="stWidgetLabel"] p {{ font-size: 0.8rem; color: {MUTED}; }}
[data-testid="stCaptionContainer"] p {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {MUTED};
}}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{ font-family: 'JetBrains Mono', monospace; }}
.stDownloadButton > button {{
    background: transparent;
    color: {TEXT};
    border: 1px solid {LINE};
    border-radius: 2px;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.9rem;
}}
.stDownloadButton > button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
    background: transparent;
}}
[data-testid="stExpander"] {{ border: 1px solid {LINE}; border-radius: 2px; background: {PANEL_SOFT}; }}
[data-testid="stExpander"] summary {{ font-size: 0.8rem; color: {TEXT}; }}
[data-testid="stImage"] img {{ border: 1px solid {LINE}; }}

@media (max-width: 780px) {{
    .block-container {{ padding: 1.4rem 0.9rem 2rem 0.9rem; }}
    .bar {{ grid-template-columns: 96px 1fr 26px; }}
    .stat .v {{ font-size: 1.2rem; }}
}}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model dan inferensi
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model():
    """Muat checkpoint sekali saja, dipakai ulang lintas rerun Streamlit."""
    from ultralytics import YOLO

    return YOLO(str(MODEL_PATH))


@st.cache_data(show_spinner=False, max_entries=12)
def run_inference(image_key: str, _image: Image.Image):
    """Jalankan model pada CONF_FLOOR lalu kembalikan deteksi mentahnya.

    `image_key` (digest isi gambar) yang menjadi kunci cache; `_image` diawali
    garis bawah supaya Streamlit tidak mencoba mem-hash objek PIL-nya.
    """
    model = load_model()
    started = time.perf_counter()
    result = model.predict(_image, conf=CONF_FLOOR, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    names = result.names  # nama kelas dibaca dari checkpoint, bukan di-hardcode
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            {
                "cls_id": cls_id,
                "name": names[cls_id],
                "conf": float(box.conf[0]),
                "xyxy": (x1, y1, x2, y2),
            }
        )
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections, elapsed_ms


@st.cache_data(show_spinner=False)
def class_catalog() -> dict[int, str]:
    """Daftar kelas apa adanya dari checkpoint, terurut menurut indeks."""
    names = load_model().names
    return {int(i): names[i] for i in sorted(names)}


@st.cache_data(show_spinner=False)
def headline_metrics() -> dict[str, dict[str, float]]:
    """Baca angka evaluasi dari metrics/metrics_summary.csv.

    Angkanya sengaja tidak ditulis ulang di kode: README, isi repo, dan
    tampilan aplikasi membaca sumber yang sama sehingga tidak bisa melenceng.
    Kalau berkasnya tidak ada, aplikasi tetap jalan tanpa baris metrik.
    """
    if not METRICS_PATH.exists():
        return {}
    parsed: dict[str, dict[str, float]] = {}
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                parsed[row["metric"]] = {
                    "valid": float(row["valid"]),
                    "test": float(row["test"]),
                }
            except (KeyError, TypeError, ValueError):
                continue  # lewati baris yang tidak berbentuk angka
    return parsed


# ---------------------------------------------------------------------------
# Penggambaran overlay deteksi
# ---------------------------------------------------------------------------

DISPLAY_MIN = 900   # citra latih hanya 300px; perbesar agar overlay tetap tajam
DISPLAY_MAX = 1400  # dan batasi unggahan besar agar render tetap ringan


@st.cache_resource(show_spinner=False)
def _font_files() -> tuple[str | None, str | None]:
    """DejaVu ikut terpasang bersama matplotlib (dependensi ultralytics)."""
    try:
        import matplotlib

        ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        sans, mono = ttf / "DejaVuSans-Bold.ttf", ttf / "DejaVuSansMono-Bold.ttf"
        if sans.exists() and mono.exists():
            return str(sans), str(mono)
    except Exception:
        pass
    return None, None


def _font(path: str | None, size: int):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow lawas
        return ImageFont.load_default()


def _display_canvas(image: Image.Image) -> tuple[Image.Image, float]:
    width, height = image.size
    longest = max(width, height)
    if longest < DISPLAY_MIN:
        scale = DISPLAY_MIN / longest
    elif longest > DISPLAY_MAX:
        scale = DISPLAY_MAX / longest
    else:
        return image.copy(), 1.0
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return image.resize((round(width * scale), round(height * scale)), resample), scale


def draw_overlay(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Gambar kotak bersudut siku, label kelas, dan skor confidence."""
    canvas, scale = _display_canvas(image)
    if not detections:
        return canvas

    base = canvas.convert("RGBA")
    width, height = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    unit = min(width, height)
    stroke = max(2, round(unit / 320))
    size = max(13, round(unit / 42))
    pad_x, pad_y, gap = round(size * 0.5), round(size * 0.34), round(size * 0.55)
    tab_h = size + pad_y * 2 + round(size * 0.25)

    font_name = _font(_font_files()[0], size)
    font_conf = _font(_font_files()[1], size)

    accent, ink = _rgb(ACCENT), _rgb(INK)

    # Objek terbesar digambar lebih dulu supaya label objek kecil tidak tertimpa.
    ordered = sorted(
        detections,
        key=lambda d: (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]),
        reverse=True,
    )

    for det in ordered:
        x1, y1, x2, y2 = (v * scale for v in det["xyxy"])
        x1, x2 = max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))
        y1, y2 = max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))

        # garis tipis mengelilingi kotak, lalu siku tebal di keempat sudut
        draw.rectangle([x1, y1, x2, y2], outline=accent + (85,), width=stroke)
        arm = max(stroke * 4, min(round(min(x2 - x1, y2 - y1) * 0.24), round(unit * 0.06)))
        for cx, dx in ((x1, 1), (x2, -1)):
            for cy, dy in ((y1, 1), (y2, -1)):
                draw.line([(cx, cy), (cx + dx * arm, cy)], fill=accent + (255,), width=stroke * 2)
                draw.line([(cx, cy), (cx, cy + dy * arm)], fill=accent + (255,), width=stroke * 2)

        label, score = det["name"], f"{det['conf']:.2f}"
        w_label = draw.textlength(label, font=font_name)
        w_score = draw.textlength(score, font=font_conf)
        tab_w = w_label + w_score + gap + pad_x * 2

        tx = min(max(0.0, x1), max(0.0, width - tab_w))
        ty = y1 - tab_h
        if ty < 0:  # kotak menempel tepi atas -> label dipindah ke dalam kotak
            ty = min(y1, max(0.0, height - tab_h))

        draw.rectangle([tx, ty, tx + tab_w, ty + tab_h], fill=accent + (242,))
        mid = ty + tab_h / 2
        draw.text((tx + pad_x, mid), label, font=font_name, fill=ink + (255,), anchor="lm")
        draw.text(
            (tx + pad_x + w_label + gap, mid),
            score,
            font=font_conf,
            fill=ink + (180,),
            anchor="lm",
        )

    return Image.alpha_composite(base, layer).convert("RGB")


# ---------------------------------------------------------------------------
# Potongan tampilan kecil
# ---------------------------------------------------------------------------


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="sec"><span class="t">{title}</span>'
        f'<span class="n">{note}</span></div>',
        unsafe_allow_html=True,
    )


def empty_state(mark: str, message: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="mark">{mark}</div>'
        f'<div class="msg">{message}</div></div>',
        unsafe_allow_html=True,
    )


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Kepala halaman
# ---------------------------------------------------------------------------

catalog = class_catalog()

st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">YOLO11 Object Detection</div>
        <h1>Urban Object Inspection Panel</h1>
        <p>YOLO11 fine-tuned model for recognizing 26 classes of street objects and
        urban infrastructure. Upload an image or select one of the samples, then adjust the
        confidence threshold to see which objects the model finds along with their scores.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = headline_metrics()
primary = metrics.get("mAP@0.5:0.95 (primary)")
map_spec = (
    f'<div class="spec"><span class="k">Test mAP@0.5:0.95</span>'
    f'<span class="v">{primary["test"]:.3f}</span></div>'
    if primary
    else ""
)

st.markdown(
    f"""
    <div class="specbar">
        <div class="spec"><span class="k">Architecture</span><span class="v">YOLO11</span></div>
        <div class="spec"><span class="k">Number of classes</span><span class="v">{len(catalog)}</span></div>
        <div class="spec"><span class="k">Training resolution</span><span class="v">{TRAIN_RES}</span></div>
        {map_spec}
        <div class="spec"><span class="k">Device</span><span class="v">CPU</span></div>
        <div class="spec"><span class="k">Weight</span><span class="v">best.pt</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.32], gap="medium")

# ---------------------------------------------------------------------------
# Panel kiri -- sumber gambar dan kontrol
# ---------------------------------------------------------------------------

image: Image.Image | None = None
source_label = ""

with left:
    section("Image source")

    mode = st.radio(
        "Input mode",
        ["Upload image", "Select sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Upload image":
        upload = st.file_uploader(
            "Image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if upload is not None:
            # Koreksi orientasi EXIF supaya foto dari ponsel tidak tampil miring.
            image = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
            source_label = upload.name
    else:
        choice = st.selectbox("Sample image", list(SAMPLES), label_visibility="collapsed")
        image = Image.open(SAMPLES_DIR / SAMPLES[choice]).convert("RGB")
        source_label = SAMPLES[choice]
        st.image(image, use_container_width=True)

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    section("Confidence threshold")
    conf = st.slider(
        "Confidence threshold",
        min_value=CONF_FLOOR,
        max_value=0.95,
        value=CONF_DEFAULT,
        step=0.05,
        label_visibility="collapsed",
    )
    st.caption(f"Showing detections with confidence ≥ {conf:.2f}")

    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
    section("Recognized classes", f"{len(catalog)} classes")
    chips = "".join(
        f'<span class="chip"><span class="i">{idx:02d}</span>'
        f'<span class="n">{name}</span></span>'
        for idx, name in catalog.items()
    )
    st.markdown(f'<div class="classes">{chips}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note">The model was trained on <b>only the {len(catalog)} classes above</b>.
        Anything outside that list generally goes <b>undetected</b> — and when it is detected, it is
        usually <b>misclassified</b> as the nearest similar-looking class: a van reads as Truck,
        a lamp post as Branch.</div>
        <div class="note">Every training image was <b>{TRAIN_RES} pixels</b>. In high-resolution photos,
        small or distant objects — traffic lights, traffic cones, fire hydrants — are often missed,
        because their scale differs so much from the training data. Lowering the confidence
        threshold recovers some of them, at the cost of more false positives.</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Panel kanan -- hasil deteksi
# ---------------------------------------------------------------------------

with right:
    if image is None:
        section("Detection results")
        empty_state(
            "Waiting for input",
            "No image has been loaded. Upload a JPG, JPEG, or PNG file via the left "
            "panel, or switch to “Select sample” mode to try one of the "
            "included test images.",
        )
    else:
        with st.spinner("Running model…"):
            key = hashlib.sha1(image.tobytes()).hexdigest()
            all_dets, elapsed_ms = run_inference(key, image)

        dets = [d for d in all_dets if d["conf"] >= conf]

        counts: dict[str, int] = {}
        for det in dets:
            counts[det["name"]] = counts.get(det["name"], 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        section("Detection results", source_label)
        annotated = draw_overlay(image, dets)
        st.image(annotated, use_container_width=True)

        st.markdown(
            f"""
            <div class="stats">
                <div class="stat"><span class="k">Detected objects</span>
                    <span class="v">{len(dets)}</span></div>
                <div class="stat"><span class="k">Distinct classes</span>
                    <span class="v">{len(ranked)}</span></div>
                <div class="stat"><span class="k">Inference time</span>
                    <span class="v">{elapsed_ms:.0f}</span><span class="u">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not dets:
            pending = len(all_dets)
            hint = (
                f"There are {pending} candidates below this threshold — lower the "
                "confidence slider in the left panel to show them."
                if pending
                else "The model found no candidates at all, even at the lowest "
                f"threshold ({CONF_FLOOR:.2f}). It’s likely this image doesn’t contain "
                f"objects from any of the {len(catalog)} supported classes."
            )
            empty_state(
                "No objects",
                f"No objects with confidence ≥ {conf:.2f}. {hint}",
            )
        else:
            section("Count per class")
            top = ranked[0][1]
            rows = "".join(
                '<div class="bar">'
                f'<span class="name">{name}</span>'
                f'<span class="track"><span class="fill" style="width:{count / top * 100:.0f}%"></span></span>'
                f'<span class="num">{count}</span>'
                "</div>"
                for name, count in ranked
            )
            st.markdown(rows, unsafe_allow_html=True)

            st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
            with st.expander(f"Detection details ({len(dets)})"):
                detail = "".join(
                    '<div class="det">'
                    f'<span class="i">{i:02d}</span>'
                    f"<span>{det['name']}</span>"
                    f'<span class="c">{det["conf"]:.2f}</span>'
                    "</div>"
                    for i, det in enumerate(dets, start=1)
                )
                st.markdown(detail, unsafe_allow_html=True)

            st.markdown('<div class="gap"></div>', unsafe_allow_html=True)
            st.download_button(
                "Download annotated image",
                data=png_bytes(annotated),
                file_name=f"deteksi_{Path(source_label).stem or 'gambar'}.png",
                mime="image/png",
            )

st.markdown(
    f"""
    <div class="foot">Confidence scores are raw model outputs, not ground truth. Inferensi
    runs once per image at {CONF_FLOOR:.2f}; sliding the threshold only filters existing
    results, ensuring instant response.</div>
    """,
    unsafe_allow_html=True,
)
