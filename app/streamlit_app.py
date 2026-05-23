from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

try:
    import moviepy.editor as moviepy
except ImportError:
    moviepy = None

from app.inference import MODEL_REGISTRY, load_torchvision, load_yolo, predict_image_torchvision, resolve_weights

ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = ROOT / "temp_videos"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="Детекция пожара — демо")
st.title("Детекция лесного пожара (БПЛА)")

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "locations" not in st.session_state:
    st.session_state.locations = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_kind" not in st.session_state:
    st.session_state.model_kind = "yolo"
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None


def create_map(df_points: pd.DataFrame, df_trajectory: pd.DataFrame) -> None:
    path_layer = pdk.Layer(
        "PathLayer",
        data=df_trajectory,
        get_path="path",
        get_color=[255, 80, 0],
        width_scale=20,
        width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )
    scatterplot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position="[lon, lat]",
        get_color="[255, 80, 0, 180]",
        get_radius=400,
        pickable=True,
        auto_highlight=True,
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=df_points,
        get_position="[lon, lat]",
        get_text="name",
        get_color="[255, 255, 255, 255]",
        get_size=14,
        get_alignment_baseline="'top'",
    )
    view_state = pdk.ViewState(
        latitude=float(df_points["lat"].mean()),
        longitude=float(df_points["lon"].mean()),
        zoom=10,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[path_layer, scatterplot_layer, text_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v10",
        )
    )


def init_locations() -> None:
    locations = [
        {"name": "Точка облёта 1", "lat": 54.921094, "lon": 61.192561, "video_path": ""},
        {"name": "Точка облёта 2", "lat": 54.981479, "lon": 61.111365, "video_path": ""},
    ]
    df = pd.DataFrame(locations)
    trajectory = [
        {
            "name": "Маршрут БПЛА",
            "path": [
                [61.363556, 54.980254],
                [61.111365, 54.981479],
                [61.192561, 54.921094],
                [61.363556, 54.980254],
            ],
        }
    ]
    st.session_state.locations = (df, pd.DataFrame(trajectory))


with st.sidebar:
    st.header("Настройки")
    model_label = st.selectbox("Архитектура", list(MODEL_REGISTRY.keys()))
    conf = st.slider("Порог уверенности", 0.01, 0.9, 0.25, 0.01)
    mode = st.radio("Режим", ["Видео", "Изображение"])

cfg = MODEL_REGISTRY[model_label]

if st.session_state.model is None or st.session_state.get("selected_model") != model_label:
    with st.spinner("Загрузка модели..."):
        if cfg["type"] == "yolo":
            weights = resolve_weights(cfg["default_weights"])
            st.session_state.model = load_yolo(weights, cfg["fallback"])
            st.session_state.model_kind = "yolo"
        else:
            weights = resolve_weights(cfg["default_weights"])
            st.session_state.model = load_torchvision(cfg["arch"], weights)
            st.session_state.model_kind = "torchvision"
        st.session_state.selected_model = model_label

if st.session_state.locations is None:
    init_locations()

if mode == "Изображение":
    img_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "webp"])
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Исходное")
            st.image(image, use_container_width=True)
        with col_b:
            st.subheader("Детекция")
            if st.session_state.model_kind == "yolo":
                results = st.session_state.model.predict(image, conf=conf, verbose=False)
                st.image(results[0].plot(), use_container_width=True)
            else:
                boxes, scores = predict_image_torchvision(st.session_state.model, image, conf=conf)
                drawn = image.copy()
                draw = ImageDraw.Draw(drawn)
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.tolist()
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1), f"fire {score:.2f}", fill="red")
                st.image(drawn, use_container_width=True)
else:
    uploaded_file = st.file_uploader(
        "Загрузите видео...", type=["mp4", "mov", "avi", "asf", "m4v"]
    )

    if uploaded_file is not None and not st.session_state.video_processed:
        if st.session_state.model_kind != "yolo":
            st.warning("Обработка видео поддерживается для YOLO. Выберите YOLOv8n или YOLOv8s.")
        else:
            with st.spinner("Обработка видео..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, dir=TEMP_DIR, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                tfile.close()

                proc_name = f"{Path(video_path).stem}_processed"
                st.session_state.model.predict(
                    source=video_path,
                    conf=conf,
                    hide_conf=True,
                    max_det=10,
                    save=True,
                    project=str(TEMP_DIR),
                    name=proc_name,
                )

                clear_name = Path(video_path).name
                output_avi = TEMP_DIR / proc_name / clear_name.replace(".mp4", ".avi")
                output_mp4 = output_avi.with_suffix(".mp4")

                if output_avi.exists() and moviepy is not None:
                    clip = moviepy.VideoFileClip(str(output_avi))
                    clip.write_videofile(str(output_mp4), logger=None)
                elif output_avi.exists():
                    output_mp4 = output_avi

                st.session_state.processed_video = str(output_mp4)
                df_pts, df_traj = st.session_state.locations
                df_pts = df_pts.copy()
                df_pts.loc[1, "video_path"] = str(output_mp4)
                st.session_state.locations = (df_pts, df_traj)
                st.session_state.video_processed = True
                st.success("Видео обработано.")

    if st.session_state.video_processed:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("Маршрут БПЛА")
            create_map(st.session_state.locations[0], st.session_state.locations[1])
        with col2:
            st.header("Результат")
            df_pts = st.session_state.locations[0]
            for _, row in df_pts.iterrows():
                if row["video_path"] and Path(row["video_path"]).exists():
                    if st.button(f"Воспроизвести: {row['name']}"):
                        st.video(row["video_path"])
            if st.session_state.processed_video and Path(st.session_state.processed_video).exists():
                st.video(st.session_state.processed_video)

    if st.button("Сбросить сессию"):
        for key in ("video_processed", "model", "locations", "processed_video", "selected_model"):
            st.session_state.pop(key, None)
        st.rerun()
