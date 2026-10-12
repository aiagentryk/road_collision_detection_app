
import streamlit as st
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from ultralytics import YOLO


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RoadGuard AI | Collision Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM DASHBOARD DESIGN
# ==================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            145deg, #080F1D 0%, #101B2D 55%, #0B1220 100%
        );
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] {
        background: #0B1220;
        border-right: 1px solid #253247;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 15, 29, 0.8);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(110deg, #17263D, #101827);
        border: 1px solid #2B3C55;
        border-radius: 18px;
        padding: 25px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.20);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: 0.3px;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #9FB3CC;
        margin-top: 8px;
    }

    .hero-tag {
        display: inline-block;
        margin-top: 14px;
        background: #123B36;
        color: #6EE7B7;
        border: 1px solid #246A5B;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 700;
    }

    .section-title {
        color: #F1F5F9;
        font-size: 21px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: linear-gradient(145deg, #17243A, #111B2D);
        border: 1px solid #2A3A52;
        border-radius: 14px;
        padding: 18px;
        min-height: 112px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #A8B8CC;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-blue {
        border-left: 4px solid #60A5FA;
    }

    .metric-red {
        border-left: 4px solid #F87171;
    }

    .metric-green {
        border-left: 4px solid #34D399;
    }

    .metric-yellow {
        border-left: 4px solid #FBBF24;
    }

    .panel {
        background: #111C2E;
        border: 1px solid #2B3C55;
        border-radius: 14px;
        padding: 17px;
        margin-bottom: 14px;
    }

    .panel-heading {
        color: #F1F5F9;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .muted {
        color: #9FB0C5;
        font-size: 13px;
    }

    .safe-banner {
        background: #10352F;
        color: #A7F3D0;
        border: 1px solid #216B59;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0 14px 0;
        font-weight: 600;
    }

    .alert-banner {
        background: #401A23;
        color: #FECACA;
        border: 1px solid #8E3545;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0 14px 0;
        font-weight: 700;
    }

    .unknown-banner {
        background: #30291A;
        color: #FDE68A;
        border: 1px solid #75602D;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0 14px 0;
    }

    .footer {
        color: #8495AB;
        text-align: center;
        font-size: 12px;
        border-top: 1px solid #27364B;
        padding-top: 18px;
        margin-top: 28px;
    }

    div[data-testid="stFileUploader"] {
        background: #111C2E;
        border: 1px dashed #49617E;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #1D4ED8;
        color: white;
    }

    [data-testid="stMetric"] {
        background: #111C2E;
        border: 1px solid #2B3C55;
        border-radius: 12px;
        padding: 12px;
    }

    h1, h2, h3 {
        color: #F1F5F9;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">ROADGUARD AI</div>
    <div class="hero-subtitle">
        ROAD COLLISION DETECTION USING DISTANCE MEASUREMENT
        <br>
        Intelligent road-object detection and distance-based
        proximity monitoring
    </div>
    <div class="hero-tag">● AI DETECTION DASHBOARD</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD MODELS
# ==================================================

@st.cache_resource
def load_detection_model():
    return YOLO("best.pt")


@st.cache_resource
def load_depth_model():
    return YOLO("yolo26n-depth.pt")


try:
    detector = load_detection_model()
    depth_model = load_depth_model()

except Exception as error:
    st.error(f"Unable to load the AI models: {error}")
    st.info(
        "Check that best.pt and yolo26n-depth.pt are located "
        "in the same folder as app.py."
    )
    st.stop()


# ==================================================
# 5. SIDEBAR SETTINGS
# ==================================================

st.sidebar.markdown("## ROADGUARD AI")
st.sidebar.caption("SYSTEM CONFIGURATION")
st.sidebar.divider()

st.sidebar.markdown("### Detection Settings")

CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

COLLISION_DISTANCE_THRESHOLD = st.sidebar.slider(
    "Alert distance (metres)",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

st.sidebar.divider()

st.sidebar.markdown("### AI Components")

st.sidebar.markdown("""
- **Object detection:** `best.pt`
- **Depth estimation:** `yolo26n-depth.pt`
- **Input:** Road image
- **Alert:** Distance threshold
""")

st.sidebar.divider()

st.sidebar.caption(
    "Prototype for educational and research use. "
    "Not intended as a vehicle safety control."
)


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def estimate_object_distance(depth_map, box):
    """Estimate distance using the median depth near the box centre."""

    if depth_map is None or depth_map.ndim != 2:
        return None

    height, width = depth_map.shape

    x1, y1, x2, y2 = map(int, box)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    center_x = max(0, min(center_x, width - 1))
    center_y = max(0, min(center_y, height - 1))

    radius = 3

    x_start = max(0, center_x - radius)
    x_end = min(width, center_x + radius + 1)

    y_start = max(0, center_y - radius)
    y_end = min(height, center_y + radius + 1)

    region = depth_map[y_start:y_end, x_start:x_end]

    valid_values = region[
        np.isfinite(region) & (region > 0)
    ]

    if valid_values.size == 0:
        return None

    return float(np.median(valid_values))


def get_risk(distance):
    """Classify proximity using the selected distance threshold."""

    if distance is None:
        return "Unknown"

    if distance <= COLLISION_DISTANCE_THRESHOLD:
        return "COLLISION ALERT"

    return "No Threshold Alert"


def get_box_color(status):
    """Return an OpenCV BGR colour for each status."""

    if status == "COLLISION ALERT":
        return (0, 0, 255)

    if status == "Unknown":
        return (0, 190, 255)

    return (40, 190, 100)


def draw_label(image, text, x, y, color):
    """Draw a readable label above a detection box."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2

    text_width, text_height = cv2.getTextSize(
        text, font, font_scale, thickness
    )[0]

    x = max(0, min(x, image.shape[1] - text_width - 10))
    y = max(text_height + 10, y)

    cv2.rectangle(
        image,
        (x, y - text_height - 8),
        (x + text_width + 8, y + 4),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ==================================================
# 7. IMAGE UPLOAD PANEL
# ==================================================

st.markdown(
    '<div class="section-title">01 &nbsp; Image Analysis</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="panel">
    <div class="panel-heading">Upload Road Image</div>
    <div class="muted">
        Supported formats: JPG, JPEG and PNG.
        Upload an image to detect road objects and estimate their distances.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"],
    help="Select a road image for object detection and distance estimation."
)


# ==================================================
# 8. PROCESS UPLOADED IMAGE
# ==================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        with st.spinner("Running object detection and depth estimation..."):

            # Object detection
            detection_result = detector.predict(
                image_array,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )[0]

            # Depth estimation
            depth_result = depth_model.predict(
                image_array,
                verbose=False
            )[0]

        # ------------------------------------------
        # EXTRACT DEPTH MAP
        # ------------------------------------------

        depth_map = None

        if getattr(depth_result, "depth", None) is not None:

            depth_data = depth_result.depth.data

            if hasattr(depth_data, "cpu"):
                depth_map = depth_data.cpu().numpy()
            else:
                depth_map = np.asarray(depth_data)

            depth_map = np.squeeze(depth_map)

            if depth_map.ndim != 2:
                depth_map = None
                st.warning(
                    "A valid two-dimensional depth map was not returned."
                )

        else:
            st.warning("The depth model did not return a depth map.")

        # ------------------------------------------
        # PREPARE ANNOTATED IMAGE
        # ------------------------------------------

        annotated_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        detections = []

        # ------------------------------------------
        # PROCESS DETECTED OBJECTS
        # ------------------------------------------

        if detection_result.boxes is not None:

            original_height, original_width = image_array.shape[:2]

            if depth_map is not None:
                depth_height, depth_width = depth_map.shape

                scale_x = depth_width / original_width
                scale_y = depth_height / original_height

            for box in detection_result.boxes:

                coordinates = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                x1, y1, x2, y2 = coordinates

                confidence = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())

                class_name = detection_result.names.get(
                    class_id,
                    str(class_id)
                )

                # Estimate distance
                distance = None

                if depth_map is not None:

                    depth_box = (
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y
                    )

                    distance = estimate_object_distance(
                        depth_map,
                        depth_box
                    )

                status = get_risk(distance)
                box_color = get_box_color(status)

                # Draw bounding box
                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Prepare label
                if distance is not None:
                    distance_text = f"{distance:.1f} m"
                else:
                    distance_text = "Distance unknown"

                label = (
                    f"{class_name} | {confidence:.2f} | "
                    f"{distance_text}"
                )

                draw_label(
                    annotated_image,
                    label,
                    x1,
                    max(y1, 25),
                    box_color
                )

                detections.append({
                    "Object": class_name,
                    "Confidence": round(confidence, 3),
                    "Estimated distance (m)": (
                        round(distance, 2)
                        if distance is not None
                        else None
                    ),
                    "Proximity status": status
                })

        # Convert annotated image to RGB
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        # ==========================================
        # 9. IMAGE COMPARISON
        # ==========================================

        st.markdown(
            '<div class="section-title">02 &nbsp; Detection Results</div>',
            unsafe_allow_html=True
        )

        original_col, result_col = st.columns(2, gap="large")

        with original_col:
            st.markdown(
                '<div class="panel-heading">Original Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                use_container_width=True
            )

        with result_col:
            st.markdown(
                '<div class="panel-heading">AI Detection Output</div>',
                unsafe_allow_html=True
            )

            st.image(
                annotated_image,
                use_container_width=True
            )

        # ==========================================
        # 10. SUMMARY METRICS
        # ==========================================

        st.markdown(
            '<div class="section-title">03 &nbsp; Safety Overview</div>',
            unsafe_allow_html=True
        )

        alert_objects = [
            item for item in detections
            if item["Proximity status"] == "COLLISION ALERT"
        ]

        unknown_objects = [
            item for item in detections
            if item["Proximity status"] == "Unknown"
        ]

        known_distances = [
            item["Estimated distance (m)"]
            for item in detections
            if item["Estimated distance (m)"] is not None
        ]

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.markdown(f"""
            <div class="metric-card metric-blue">
                <div class="metric-label">Objects Detected</div>
                <div class="metric-value">{len(detections)}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric2:
            st.markdown(f"""
            <div class="metric-card metric-red">
                <div class="metric-label">Proximity Alerts</div>
                <div class="metric-value">{len(alert_objects)}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric3:
            st.markdown(f"""
            <div class="metric-card metric-yellow">
                <div class="metric-label">Unknown Distances</div>
                <div class="metric-value">{len(unknown_objects)}</div>
            </div>
            """, unsafe_allow_html=True)

        with metric4:
            st.markdown(f"""
            <div class="metric-card metric-green">
                <div class="metric-label">Alert Threshold</div>
                <div class="metric-value">{COLLISION_DISTANCE_THRESHOLD} m</div>
            </div>
            """, unsafe_allow_html=True)

        # ==========================================
        # 11. PROXIMITY ALERT PANEL
        # ==========================================

        st.markdown(
            '<div class="section-title">04 &nbsp; Proximity Monitoring</div>',
            unsafe_allow_html=True
        )

        if alert_objects:

            st.markdown(
                f"""
                <div class="alert-banner">
                    COLLISION DISTANCE ALERT
                    <br>
                    {len(alert_objects)} detected object(s) are
                    at or below the configured threshold of
                    {COLLISION_DISTANCE_THRESHOLD} metres.
                </div>
                """,
                unsafe_allow_html=True
            )

            for item in alert_objects:
                st.error(
                    f"{item['Object']} — estimated distance: "
                    f"{item['Estimated distance (m)']} m"
                )

        else:

            st.markdown(
                f"""
                <div class="safe-banner">
                    NO THRESHOLD ALERT
                    <br>
                    No available distance estimate is at or below
                    {COLLISION_DISTANCE_THRESHOLD} metres.
                </div>
                """,
                unsafe_allow_html=True
            )

        if unknown_objects:

            st.markdown(
                f"""
                <div class="unknown-banner">
                    DISTANCE UNAVAILABLE
                    <br>
                    {len(unknown_objects)} detected object(s) have
                    no usable distance estimate. Their proximity
                    status cannot be determined.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ==========================================
        # 12. DETECTION DATA TABLE
        # ==========================================

        st.markdown(
            '<div class="section-title">05 &nbsp; Detailed Detection Report</div>',
            unsafe_allow_html=True
        )

        if detections:

            results_df = pd.DataFrame(detections)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Object": st.column_config.TextColumn(
                        "Detected Object"
                    ),
                    "Confidence": st.column_config.NumberColumn(
                        "Confidence",
                        format="%.3f"
                    ),
                    "Estimated distance (m)": st.column_config.NumberColumn(
                        "Estimated Distance (m)",
                        format="%.2f"
                    ),
                    "Proximity status": st.column_config.TextColumn(
                        "Proximity Status"
                    )
                }
            )

            # Download report
            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Detection Report (CSV)",
                data=csv_data,
                file_name="road_detection_report.csv",
                mime="text/csv"
            )

            if known_distances:
                st.caption(
                    f"Closest estimated object distance: "
                    f"{min(known_distances):.2f} m"
                )

        else:

            st.info(
                "No objects were detected. Try another image or "
                "reduce the detection confidence threshold."
            )

    except Exception as error:

        st.error(f"Image processing failed: {error}")

        st.info(
            "Check the model files, package versions, and depth "
            "model output format."
        )

else:

    st.markdown("""
    <div class="panel">
        <div class="panel-heading">How the System Works</div>
        <p class="muted">
            1. Upload a road image.
            <br><br>
            2. Detect vehicles, pedestrians and cyclists.
            <br><br>
            3. Estimate distances using the depth model.
            <br><br>
            4. Compare available estimates with the selected threshold.
            <br><br>
            5. Review annotated images, alerts and the detection report.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 13. FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    <strong>ROADGUARD AI</strong>
    <br>
    Road Collision Detection Using Distance Measurement
    <br><br>
    YOLO26n Object Detection | Depth Estimation | Python | Streamlit
    <br>
    Educational prototype — not a validated collision-warning system.
</div>
""", unsafe_allow_html=True)
