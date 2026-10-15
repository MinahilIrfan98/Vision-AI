import streamlit as st
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import pandas as pd
import os
import altair as alt
from main import detect_phone_usage

# Page Config
st.set_page_config(
    page_title="VisionAI",
    page_icon="page_icon.png",
    layout="wide"
)

# Color Palette
ACCENT_BLUE = "#2F6BFF"
BG_NAVY = "#0B1220"
CARD_NAVY = "#101A2E"
BORDER_COLOR = "rgba(255,255,255,0.06)"
TEXT_MAIN = "#FFFFFF"
TEXT_DIM = "#94a3b8"

# Object Color Mapping (matches main.py BGR)
# BGR (0,255,0) -> RGB (0,255,0) Green
# BGR (0,0,255) -> RGB (255,0,0) Red
CLASS_COLORS = {
    "person": "#00FF00",
    "cell phone": "#FF0000",
    "laptop": "#3B82F6",
    "cup": "#F59E0B",
    "potted plant": "#10B981",
    "dining table": "#EC4899",
    "vase": "#A78BFA",
    "other": "#06B6D4",
}

def class_color(class_name):
    """Return a stable, distinct color for each detected class."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    palette = ["#F97316", "#14B8A6", "#EAB308", "#8B5CF6", "#EF4444", "#22C55E", "#06B6D4", "#D946EF"]
    # Stable across app reruns and Python processes (unlike the built-in hash()).
    color_index = sum((index + 1) * ord(char) for index, char in enumerate(class_name)) % len(palette)
    return palette[color_index]

# Custom CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Base Page Styling */
    .stApp {{
        background-color: {BG_NAVY};
        color: {TEXT_MAIN};
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        max-width: 1440px;
        padding: 2.25rem 2.5rem 4rem;
    }}

    [data-testid="stHorizontalBlock"] {{
        align-items: flex-start;
    }}

    /* Hide default Streamlit elements */
    header, footer, #MainMenu {{
        visibility: hidden;
    }}

    /* Main layout adjustments */
    [data-testid="stVerticalBlock"] {{
        gap: 1rem;
    }}

    /* Custom Card Style */
    .custom-card {{
        background: linear-gradient(145deg, #111d33, #0e1728);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 1.35rem;
        margin-bottom: 1.1rem;
        color: {TEXT_MAIN};
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.14);
    }}

    /* Wordmark Styling */
    .wordmark {{
        font-size: 30px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        color: {TEXT_MAIN};
    }}
    .wordmark-ai {{
        color: {ACCENT_BLUE};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* Upload Zone styling */
    .upload-zone {{
        border: 1px dashed rgba(148, 163, 184, 0.32);
        border-radius: 16px;
        padding: 2.25rem 1.5rem;
        text-align: center;
        color: {TEXT_DIM};
        background: rgba(16, 26, 46, 0.62);
        transition: border-color 0.2s ease, background 0.2s ease;
    }}
    .upload-zone:hover {{
        border-color: {ACCENT_BLUE};
        background: rgba(47, 107, 255, 0.06);
    }}

    /* Pill Chips */
    .pill-chip {{
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
        color: white;
    }}

    /* Hide original file uploader label */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {CARD_NAVY} !important;
        border: 1px dashed rgba(148, 163, 184, 0.32) !important;
        border-radius: 14px !important;
        color: {TEXT_DIM} !important;
    }}

    [data-testid="stFileUploader"] section {{
        padding: 1rem;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: {TEXT_DIM};
        font-weight: 600;
        font-size: 14px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT_BLUE} !important;
        border-bottom-color: {ACCENT_BLUE} !important;
    }}

    /* Button styling */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        min-height: 2.7rem;
    }}
    .stButton > button:first-child {{
        background-color: {ACCENT_BLUE};
        color: white;
        border: none;
    }}
    .stButton > button:hover {{
        filter: brightness(1.1);
        border: none;
    }}

    /* Metric labels */
    .metric-label {{
        font-size: 12px;
        color: {TEXT_DIM};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 700;
        color: {TEXT_MAIN};
    }}

    div[data-testid="stImage"] img {{
        border-radius: 12px;
    }}

    div[data-testid="stCheckbox"] label, div[data-testid="stSlider"] label,
    div[data-testid="stMultiSelect"] label {{
        color: #cbd5e1;
        font-weight: 500;
    }}

    hr {{ border-color: rgba(148, 163, 184, 0.14); }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('yolo26n.pt')

def get_donut_chart(stats):
    """Generates an Altair donut chart for detections."""
    persons = stats.get("persons", 0)
    phones = stats.get("phones", 0)
    others = sum(stats.get("all_detected", {}).values()) - persons - phones

    class_counts = stats.get("all_detected", {})
    df = pd.DataFrame([
        {"Category": name, "Count": count, "Color": class_color(name)}
        for name, count in class_counts.items() if count > 0
    ])

    # Filter out zero counts to avoid chart errors
    df = df[df["Count"] > 0]
    if df.empty:
        return None

    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Color", type="nominal", scale=None, legend=None),
        tooltip=["Category", "Count"]
    ).properties(
        width=200,
        height=200
    )

    return chart

def render_image_results(img, annotated_img, stats):
    """Renders the image and its associated stats in the new layout."""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    show_orig = st.checkbox("Show Original", value=False)
    display_img = img if show_orig else annotated_img

    st.image(display_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bottom Row
    col_objs, col_classes = st.columns(2)
    with col_objs:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Detected Objects</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-value">👤 {stats["persons"]}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-value">📱 {stats["phones"]}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-value">📦 {sum(stats["all_detected"].values())}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_classes:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Detected Classes</div>', unsafe_allow_html=True)

        all_detected = stats["all_detected"]
        sorted_classes = sorted(all_detected.items(), key=lambda x: x[1], reverse=True)

        chips_html = ""
        for i, (cls, count) in enumerate(sorted_classes):
            if i < 5:
                color = class_color(cls)
                chips_html += f'<div class="pill-chip" style="background-color: {color}">{cls} {count}</div>'
            else:
                chips_html += f'<div class="pill-chip" style="background-color: {CLASS_COLORS["other"]}">+ {len(sorted_classes) - 5} more</div>'
                break

        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    model = load_model()

    # Header
    st.markdown(f"""
        <div class="wordmark">
            <img src="data:image/svg+xml;base64,{base64.b64encode(open('logo.svg', 'rb').read()).decode('ascii') if os.path.exists('logo.svg') else ''}" width="28" height="28">
            <span>Vision<span class="wordmark-ai">AI</span></span>
        </div>
        <div style="color: {TEXT_DIM}; font-size: 15px; margin: 0.35rem 0 2rem;">AI-powered image analysis for people and mobile phone detection</div>
    """, unsafe_allow_html=True)

    st.markdown("### Image analysis")
    st.caption("Upload one or more images, tune detection, and download annotated results.")
    left_col, right_col = st.columns([1.8, 1], gap="large")

    with left_col:
        uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, help="Supported formats: JPG, JPEG, PNG")
        if not uploaded_files:
            st.markdown(f"""
                <div class="upload-zone">
                    <div style="font-size: 34px; margin-bottom: 0.7rem;">▧</div>
                    <h3 style="color: {TEXT_MAIN}; margin: 0 0 0.4rem;">Add images to get started</h3>
                    <p style="color: {TEXT_DIM}; margin: 0;">Drop files above or browse from your device</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.expander("Detection settings", expanded=True):
                s_col1, s_col2 = st.columns([1, 1.5])
                with s_col1:
                    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
                    show_conf = st.checkbox("Show confidence scores", value=False)
                with s_col2:
                    hide_objects = st.multiselect("Hide classes", list(model.names.values()), default=[], placeholder="Show all classes")

            for index, uploaded_file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                if img is None:
                    st.error(f"Could not read {uploaded_file.name}. Please try another image.")
                    continue

                annotated_img, stats = detect_phone_usage(img, model, conf_threshold, hide_objects, show_conf)
                update_stats(stats)
                st.markdown(f"#### {uploaded_file.name}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ann_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                render_image_results(img_rgb, ann_rgb, stats)

                is_success, buffer = cv2.imencode(".jpg", annotated_img)
                if is_success:
                    st.download_button(
                        "Download annotated image", data=buffer.tobytes(),
                        file_name=f"result_{os.path.splitext(uploaded_file.name)[0]}.jpg",
                        mime="image/jpeg", key=f"dl_{index}_{uploaded_file.name}",
                    )
                st.divider()

    with right_col:
        # Detection Summary
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Detection Summary</div>', unsafe_allow_html=True)

        if 'current_stats' in st.session_state:
            chart = get_donut_chart(st.session_state.current_stats)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown('<div style="text-align: center; padding: 2rem 1rem; color: #94a3b8;">Upload an image to see detection insights.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Detected Objects Table
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Detected Objects</div>', unsafe_allow_html=True)

        if 'current_stats' in st.session_state:
            stats = st.session_state.current_stats
            all_detected = stats["all_detected"]
            sorted_classes = sorted(all_detected.items(), key=lambda x: x[1], reverse=True)

            # Create a clean table
            table_html = '<table style="width: 100%; border-collapse: collapse; font-size: 13px; color: #FFFFFF;">'
            table_html += '<thead><tr style="text-align: left; color: #94a3b8; border-bottom: 1px solid rgba(255,255,255,0.1);"><th style="padding: 8px 0;">Class</th><th style="padding: 8px 0;">Count</th><th style="padding: 8px 0;">%</th></tr></thead><tbody>'

            total = sum(all_detected.values())
            for cls, count in sorted_classes:
                color = class_color(cls)
                perc = (count / total) * 100 if total > 0 else 0
                table_html += f'<tr><td style="padding: 8px 0;"><span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {color}; margin-right: 8px;"></span>{cls}</td><td style="padding: 8px 0;">{count}</td><td style="padding: 8px 0;">{perc:.1f}%</td></tr>'

            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 1.5rem 1rem; color: #94a3b8;">Detected classes will appear here.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Workspace</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:13px;line-height:1.6;">Analyze images privately in this session. Download each annotated result from its image card.</p>', unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def update_stats(stats):
    """Helper to update session state for the right column."""
    st.session_state.current_stats = stats

if __name__ == "__main__":
    main()
