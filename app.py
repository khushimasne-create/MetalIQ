from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# 1. CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "industrial_defect_cnn.keras"
CLASS_PATH = BASE_DIR / "outputs" / "class_names.txt"

IMAGE_SIZE = (200, 200)

st.set_page_config(
    page_title="MetalIQ | Industrial Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. WEBSITE STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background: #101a2b;
    }

    [data-testid="stSidebar"] * {
        color: #edf3ff;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #12233f, #174b72);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d6e6f7;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #172033;
        margin-top: 12px;
        margin-bottom: 14px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e0e7f0;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 3px 12px rgba(25, 45, 75, 0.04);
    }

    .metric-label {
        color: #65748b;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #172033;
        font-size: 23px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #718096;
        font-size: 12px;
        margin-top: 8px;
    }

    .panel {
        background: white;
        border: 1px solid #e0e7f0;
        border-radius: 15px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .small-muted {
        color: #718096;
        font-size: 13px;
    }

    div.stButton > button[kind="primary"] {
        background: #1769aa;
        border: none;
        border-radius: 9px;
        min-height: 46px;
        font-weight: 650;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #105486;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e7f0;
        padding: 16px;
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }

    /* MAKE IMAGE SOURCE AND UPLOAD/CAMERA TEXT BLACK */
    .stApp [data-testid="stRadio"] label,
    .stApp [data-testid="stRadio"] label p,
    .stApp [data-testid="stFileUploader"] label,
    .stApp [data-testid="stFileUploader"] label p,
    .stApp [data-testid="stFileUploader"] small,
    .stApp [data-testid="stFileUploader"] section,
    .stApp [data-testid="stFileUploader"] section *,
    .stApp [data-testid="stCameraInput"] label,
    .stApp [data-testid="stCameraInput"] label p,
    .stApp [data-testid="stCameraInput"] button,
    .stApp [data-testid="stCameraInput"] button * {
        color: #000000 !important;
    }

    /* WHITE UPLOADER AREA WITH BLACK TEXT */
    .stApp [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border-color: #b8c3d1 !important;
    }

    .stApp [data-testid="stCameraInput"] button {
        background: #ffffff !important;
        border-color: #b8c3d1 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🔬 MetalIQ")
    st.caption("INDUSTRIAL QUALITY INTELLIGENCE")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("◉  Inspection Dashboard")
    st.markdown("▤  Classification Analytics")
    st.markdown("⇩  Export Reports")

    st.divider()
    st.markdown("### Model information")
    st.write("**Architecture:** CNN")
    st.write("**Dataset:** NEU-CLS")
    st.write("**Input:** 200 × 200 grayscale")
    st.write("**Categories:** 6")

    st.divider()
    st.caption("Academic prototype • Decision support only")

# =========================================================
# 4. LOAD MODEL
# =========================================================
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not CLASS_PATH.exists():
    st.error(f"Class names file not found: {CLASS_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    return [
        line.strip()
        for line in CLASS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


try:
    model = load_model()
    class_names = load_classes()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

if len(class_names) != model.output_shape[-1]:
    st.error(
        "Model output count does not match class_names.txt. "
        "Verify the class order and model files from training."
    )
    st.stop()

# =========================================================
# 5. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Industrial Inspection Dashboard</h1>
    <p>
        AI-assisted metal surface classification, defect analytics,
        and inspection report generation.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 6. IMAGE INPUT
# =========================================================
st.markdown(
    '<div class="section-heading">01 / Image Acquisition</div>',
    unsafe_allow_html=True
)

input_col, preview_col = st.columns([1, 1.15], gap="large")

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Select inspection image")
    st.caption("Upload a metal surface image or capture one.")

    input_method = st.radio(
        "Image source",
        ["Upload image", "Camera capture"],
        horizontal=True
    )

    image_file = None

    if input_method == "Upload image":
        image_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG and PNG."
        )
    else:
        image_file = st.camera_input("Capture metal surface")

    st.markdown(
        '<p class="small-muted">'
        'For reliable results, use a clear, well-lit metal surface image.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with preview_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Image preview")

    if image_file is not None:
        try:
            image = Image.open(image_file).convert("L")
            st.image(
                image,
                caption=f"Input image • Original size: {image.width} × {image.height}",
                use_container_width=True
            )
        except Exception:
            st.error("This image could not be opened. Try another image.")
            image = None
    else:
        image = None
        st.info("Your selected metal surface image will appear here.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 7. RUN INSPECTION
# =========================================================
st.markdown(
    '<div class="section-heading">02 / AI Inspection</div>',
    unsafe_allow_html=True
)

analyze = st.button(
    "Run Defect Analysis",
    type="primary",
    use_container_width=True,
    disabled=image is None
)

if analyze and image is not None:
    try:
        resized = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            resized,
            dtype=np.float32
        ) / 255.0

        image_array = image_array[np.newaxis, ..., np.newaxis]

        with st.spinner("Analyzing the metal surface..."):
            scores = np.asarray(
                model.predict(image_array, verbose=0)[0],
                dtype=float
            )

        if (
            len(scores) != len(class_names)
            or not np.all(np.isfinite(scores))
        ):
            st.error("The model returned invalid prediction scores.")
            st.stop()

        predicted_index = int(np.argmax(scores))
        predicted_class = class_names[predicted_index]
        confidence = float(scores[predicted_index])

        if np.all(scores >= 0) and np.isclose(
            np.sum(scores), 1.0, atol=0.02
        ):
            score_type = "Softmax output score"
        else:
            score_type = "Model output score"

        inspection_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.session_state["inspection"] = {
            "scores": scores.tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "inspection_time": inspection_time,
            "image_size": f"{image.width} × {image.height}",
            "score_type": score_type
        }

    except Exception as error:
        st.error(f"Inspection failed: {error}")

# =========================================================
# 8. DISPLAY RESULTS
# =========================================================
result = st.session_state.get("inspection")

if result is not None:
    scores = np.asarray(result["scores"], dtype=float)
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    st.divider()
    st.markdown(
        '<div class="section-heading">03 / Inspection Results</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "This model was trained to classify six defect categories. "
        "It does not have a dedicated defect-free class, so the result "
        "must not be used to certify metal as defect-free."
    )

    metric1, metric2, metric3 = st.columns(3, gap="medium")

    with metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PREDICTED CATEGORY</div>
            <div class="metric-value">
                {predicted_class.replace("_", " ").title()}
            </div>
            <div class="metric-note">
                Highest-scoring model category
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MODEL OUTPUT SCORE</div>
            <div class="metric-value">{confidence * 100:.2f}%</div>
            <div class="metric-note">
                Not a measure of defect severity
            </div>
        </div>
        """, unsafe_allow_html=True)

    with metric3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">INSPECTION TIME</div>
            <div class="metric-value" style="font-size:17px">
                {result["inspection_time"]}
            </div>
            <div class="metric-note">Local system time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown(f"""
    <div class="panel">
        <h3 style="margin-top:0">Inspection summary</h3>
        <p>
            The CNN's highest-scoring category is
            <b>{predicted_class.replace("_", " ").title()}</b>.
            Review the image and category scores before making a
            quality-control decision.
        </p>
        <p class="small-muted">
            Input image dimensions: {result["image_size"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # 9. CLASSIFICATION GRAPH
    # =====================================================
    st.markdown(
        '<div class="section-heading">04 / Classification Analytics</div>',
        unsafe_allow_html=True
    )

    graph_col, info_col = st.columns([1.65, 1], gap="large")

    sorted_indices = np.argsort(scores)
    sorted_names = [
        class_names[i].replace("_", " ").title()
        for i in sorted_indices
    ]
    sorted_scores = scores[sorted_indices] * 100

    with graph_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Defect category comparison")
        st.caption(
            "Scores are generated from the selected image. "
            "The top-scoring category appears at the top."
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bars = ax.barh(
            sorted_names,
            sorted_scores,
            height=0.62
        )

        for bar, score in zip(bars, sorted_scores):
            ax.text(
                min(score + 1.0, 96),
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}%",
                va="center",
                fontsize=9,
                color="#25344a"
            )

        ax.set_xlim(0, 100)
        ax.set_xlabel("Model output score (%)", fontsize=10)
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Classification details")

        st.write("**Top category**")
        st.write(predicted_class.replace("_", " ").title())

        st.write("**Highest output score**")
        st.write(f"{confidence * 100:.2f}%")

        st.write("**Categories evaluated**")
        st.write(str(len(class_names)))

        st.write("**Score interpretation**")
        st.caption(
            "The graph compares model outputs. Scores do not indicate "
            "the physical size, depth, or seriousness of a defect."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # 10. RESULTS TABLE AND DOWNLOAD
    # =====================================================
    st.markdown(
        '<div class="section-heading">05 / Detailed Results & Export</div>',
        unsafe_allow_html=True
    )

    results_df = pd.DataFrame({
        "Defect Category": [
            name.replace("_", " ").title()
            for name in class_names
        ],
        "Model Output Score (%)": np.round(scores * 100, 2)
    }).sort_values(
        "Model Output Score (%)",
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    report_df = results_df.copy()
    report_df["Predicted Category"] = (
        predicted_class.replace("_", " ").title()
    )
    report_df["Inspection Time"] = result["inspection_time"]
    report_df["Input Image Size"] = result["image_size"]

    csv_data = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Inspection Report (CSV)",
        data=csv_data,
        file_name="industrial_inspection_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Report contains the model's category scores and prediction. "
        "It does not contain a verified severity assessment."
    )

else:
    st.info(
        "Select an image and click **Run Defect Analysis** to generate "
        "the prediction, comparison graph, and inspection report."
    )

# =========================================================
# 11. FOOTER
# =========================================================
st.divider()
st.caption(
    "MetalIQ • Industrial Defect Detection • CNN-based academic prototype"
)