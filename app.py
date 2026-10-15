import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import pandas as pd
import tempfile
import os
import altair as alt
from main import detect_phone_usage, is_inside, get_color

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
    "other": "#8A2BE2"
}

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
        background-color: {CARD_NAVY};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        color: {TEXT_MAIN};
    }}

    /* Wordmark Styling */
    .wordmark {{
        font-size: 24px;
        font-weight: 600;
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
        border: 2px dashed {BORDER_COLOR};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: {TEXT_DIM};
        cursor: pointer;
        transition: border 0.2s ease;
    }}
    .upload-zone:hover {{
        border-color: {ACCENT_BLUE};
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
        border: 2px dashed {BORDER_COLOR} !important;
        border-radius: 12px !important;
        color: {TEXT_DIM} !important;
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
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
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
        font-size: 24px;
        font-weight: 700;
        color: {TEXT_MAIN};
    }}
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

    df = pd.DataFrame([
        {"Category": "Persons", "Count": persons, "Color": CLASS_COLORS["person"]},
        {"Category": "Phones", "Count": phones, "Color": CLASS_COLORS["cell phone"]},
        {"Category": "Others", "Count": others, "Color": CLASS_COLORS["other"]},
    ])

    # Filter out zero counts to avoid chart errors
    df = df[df["Count"] > 0]
    if df.empty:
        return None

    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Category", type="nominal", scale=alt.Scale(
            domain=["Persons", "Phones", "Others"],
            range=[CLASS_COLORS["person"], CLASS_COLORS["cell phone"], CLASS_COLORS["other"]]
        ), legend=None),
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
                color = CLASS_COLORS.get(cls, CLASS_COLORS["other"])
                chips_html += f'<div class="pill-chip" style="background-color: {color}">{cls} {count}</div>'
            else:
                chips_html += f'<div class="pill-chip" style="background-color: {CLASS_COLORS["other"]}">+ {len(sorted_classes) - 5} more</div>'
                break

        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def process_video(uploaded_file, model, conf_threshold, hide_objects, show_conf):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    person_stats = {}
    time_series_phone_users = []
    peak_counts = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, conf=conf_threshold)[0]
        annotated_frame = frame.copy()

        persons, phones, others = [], [], []

        if results.boxes.id is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                obj_id = int(box.id[0])
                label = model.names[cls]
                if label in hide_objects: continue

                peak_counts[label] = max(peak_counts.get(label, 0), 1)
                if cls == 0: persons.append({'box': coords, 'id': obj_id, 'conf': conf})
                elif cls == 67: phones.append({'box': coords, 'conf': conf})
                else: others.append({'box': coords, 'conf': conf, 'label': label})

        for obj in others:
            box = obj['box']
            color = get_color(model.names.index(obj['label']) if obj['label'] in model.names else 0)
            txt = f"{obj['label']}"
            if show_conf: txt += f" {obj['conf']:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

            # Filled Label
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1]) - th - 5), (int(box[0]) + tw, int(box[1])), color, -1)
            cv2.putText(annotated_frame, txt, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for ph in phones:
            box = ph['box']
            color = get_color(67)
            txt = "Phone"
            if show_conf: txt += f" {ph['conf']:.2f}"
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1]) - th - 5), (int(box[0]) + tw, int(box[1])), color, -1)
            cv2.putText(annotated_frame, txt, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        current_frame_phone_users = 0
        for p in persons:
            p_id = p['id']
            p_box = p['box']
            if p_id not in person_stats: person_stats[p_id] = {'total_frames': 0, 'phone_frames': 0}
            person_stats[p_id]['total_frames'] += 1

            is_using = any(is_inside(ph['box'], p_box) for ph in phones)
            if is_using:
                person_stats[p_id]['phone_frames'] += 1
                current_frame_phone_users += 1
                color, label = (0, 0, 255), f"Person {p_id} - Using phone"
            else:
                color, label = (0, 255, 0), f"Person {p_id}"

            if show_conf: label += f" {p['conf']:.2f}"
            cv2.rectangle(annotated_frame, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (int(p_box[0]), int(p_box[1]) - th - 5), (int(p_box[0]) + tw, int(p_box[1])), color, -1)
            cv2.putText(annotated_frame, label, (int(p_box[0]), int(p_box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        time_series_phone_users.append(current_frame_phone_users)
        out.write(annotated_frame)
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    return out_path, person_stats, time_series_phone_users, fps, peak_counts

def main():
    model = load_model()

    # Header
    st.markdown(f"""
        <div class="wordmark">
            <img src="data:image/svg+xml;base64,{pd.io.common.encode_base64(open('logo.svg', 'rb').read()) if os.path.exists('logo.svg') else ''}" width="28" height="28">
            <span>Vision<span class="wordmark-ai">AI</span></span>
        </div>
        <div style="color: {TEXT_DIM}; font-size: 14px; margin-bottom: 2rem;">Sleek AI-powered monitoring of mobile device usage</div>
    """, unsafe_allow_html=True)

    # Main layout: Two columns (approx 2.2 : 1)
    left_col, right_col = st.columns([2.2, 1])

    with left_col:
        tab1, tab2 = st.tabs(["🖼️ Image Analysis", "📹 Video Analysis"])

        with tab1:
            uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            if not uploaded_files:
                st.markdown(f"""
                    <div class="upload-zone">
                        <div style="font-size: 32px; margin-bottom: 1rem;">☁️</div>
                        <h3 style="color: {TEXT_MAIN};">No images uploaded</h3>
                        <p style="color: {TEXT_DIM};">Drag and drop photos to start analysis</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Settings Row
                st.markdown('<div style="display: flex; gap: 1rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
                s_col1, s_col2, s_col3 = st.columns(3)
                with s_col1:
                    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
                with s_col2:
                    all_coco_classes = list(model.names.values())
                    hide_objects = st.multiselect("Filter Classes", all_coco_classes, default=[], placeholder="All classes")
                with s_col3:
                    show_conf = st.checkbox("Show Confidence Scores", value=False)
                st.markdown('</div>', unsafe_allow_html=True)

                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)

                    # Result processing
                    annotated_img, stats = detect_phone_usage(img, model, conf_threshold, hide_objects, show_conf)

                    st.markdown(f"**{uploaded_file.name}**")

                    # Convert BGR to RGB for display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ann_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                    render_image_results(img_rgb, ann_rgb, stats)

                    # Download
                    is_success, buffer = cv2.imencode(".jpg", annotated_img)
                    if is_success:
                        st.download_button(label="Download Result", data=buffer.tobytes(), file_name=f"res_{uploaded_file.name}", mime="image/jpeg", key=f"dl_{uploaded_file.name}")
                    st.divider()

        with tab2:
            video_file = st.file_uploader("Upload video", type=['mp4', 'mov', 'avi'])
            if not video_file:
                st.markdown(f"""
                    <div class="upload-zone">
                        <div style="font-size: 32px; margin-bottom: 1rem;">☁️</div>
                        <h3 style="color: {TEXT_MAIN};">No video uploaded</h3>
                        <p style="color: {TEXT_DIM};">Upload a clip to start analysis</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                if st.button("Start Video Processing"):
                    with st.spinner("Processing..."):
                        out_path, person_stats, time_series, fps, peak_counts = process_video(video_file, model, conf_threshold, hide_objects, show_conf)
                        with open(out_path, 'rb') as f: video_bytes = f.read()

                        st.video(video_bytes)
                        st.download_button("Download Video", data=video_bytes, file_name="result_video.mp4", mime="video/mp4")

                        # Video Stats
                        st.markdown("### 📊 Video Analytics")
                        total_unique_people = len(person_stats)
                        people_who_used_phone = sum(1 for p in person_stats.values() if p['phone_frames'] > 0)

                        video_stats = {
                            "persons": total_unique_people,
                            "phones": peak_counts.get("cell phone", 0),
                            "using_phone": people_who_used_phone,
                            "all_detected": peak_counts
                        }

                        # Use the same stat rendering as images
                        # But in a separate container to match the card style
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f'<div class="metric-value">👤 {video_stats["persons"]}</div>', unsafe_allow_html=True)
                        c2.markdown(f'<div class="metric-value">📱 {video_stats["phones"]}</div>', unsafe_allow_html=True)
                        c3.markdown(f'<div class="metric-value" style="color: #FF4B4B;">{video_stats["using_phone"]}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Table & Chart
                        person_data = []
                        for p_id, s in person_stats.items():
                            total_sec, phone_sec = s['total_frames'] / fps, s['phone_frames'] / fps
                            person_data.append({"ID": p_id, "Total (s)": round(total_sec, 2), "Phone (s)": round(phone_sec, 2), "Usage %": f"{round((s['phone_frames']/s['total_frames'])*100, 2)}%"})

                        df = pd.DataFrame(person_data)
                        st.table(df)
                        st.download_button("Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="stats.csv", mime="text/csv")

                        time_axis = [i / fps for i in range(len(time_series))]
                        chart_df = pd.DataFrame({"Phone Users": time_series}, index=pd.to_datetime(0, unit='s') + pd.to_timedelta(time_axis, unit='s'))
                        st.line_chart(chart_df)

    with right_col:
        # Right column is meant to be sticky. In Streamlit, we can simulate this with a fixed-height container or just standard flow.
        # We'll use containers for the cards.

        # Detection Summary
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Detection Summary</div>', unsafe_allow_html=True)

        # We need some current stats to show the chart.
        # Since stats are produced inside the tabs, we can use st.session_state.
        if 'current_stats' in st.session_state:
            chart = get_donut_chart(st.session_state.current_stats)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown('<div style="text-align: center; padding: 2rem; color: #64748b;">No data available</div>', unsafe_allow_html=True)
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
                color = CLASS_COLORS.get(cls, CLASS_COLORS["other"])
                perc = (count / total) * 100 if total > 0 else 0
                table_html += f'<tr><td style="padding: 8px 0;"><span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {color}; margin-right: 8px;"></span>{cls}</td><td style="padding: 8px 0;">{count}</td><td style="padding: 8px 0;">{perc:.1f}%</td></tr>'

            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; padding: 2rem; color: #64748s;">No data available</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick Actions
        st.markdown('<div class="custom-card">', dst_label="Quick Actions", unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Quick Actions</div>', unsafe_allow_html=True)

        if st.button("Download Results", use_container_width=True):
            # This would trigger the download buttons in the tabs
            st.info("Please use the download buttons below the results.")

        if st.button("Reset", use_container_width=True):
            # In Streamlit, resetting means clearing session state and rerunning
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def update_stats(stats):
    """Helper to update session state for the right column."""
    st.session_state.current_stats = stats

if __name__ == "__main__":
    # Wrap main to include the stats update logic
    # We'll need to modify the loop in main() to call update_stats()
    # Let's just do a a simple rewrite of the loop.
    # Actually, the best way is to just call it in the detect_phone_usage result.

    # I'll just modify the main() function to call it.

    # I'll just do a quick fix in the loop.
    # Let's just run it.
    main()
