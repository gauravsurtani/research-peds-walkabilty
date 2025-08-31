import os
import io
import time
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import requests
from PIL import Image
import cv2
import streamlit as st
from streamlit_folium import st_folium
import folium

# Segment Anything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


GOOGLE_STREET_VIEW_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview"
GOOGLE_STREET_VIEW_METADATA_ENDPOINT = "https://maps.googleapis.com/maps/api/streetview/metadata"

MODEL_TYPE_CHOICES = ["auto", "vit_h", "vit_l", "vit_b"]

# Common Google Places 'type' values to summarize
DEFAULT_PLACE_TYPES: List[str] = [
    "cafe",
    "restaurant",
    "school",
    "park",
    "grocery_or_supermarket",
    "pharmacy",
    "gym",
    "library",
    "transit_station",
]


def list_checkpoints_in_dir(directory: str) -> List[str]:
    if not directory:
        return []
    dirpath = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(dirpath):
        return []
    try:
        files = [
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.lower().endswith(".pth")
        ]
    except Exception:
        return []
    return sorted(files)


def infer_model_type_from_filename(filepath: str) -> Optional[str]:
    name = os.path.basename(filepath).lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"
    return None


@dataclass
class StreetViewConfig:
    size_w: int = 640
    size_h: int = 640
    fov: int = 90
    pitch: int = 0
    heading: Optional[int] = None  # if None, let API pick best


def fetch_street_view_image(api_key: str, lat: float, lon: float, cfg: StreetViewConfig, pano_id: Optional[str] = None) -> Image.Image:
    params = {
        "size": f"{cfg.size_w}x{cfg.size_h}",
        "fov": cfg.fov,
        "pitch": cfg.pitch,
        "key": api_key,
    }
    if pano_id:
        params["pano"] = pano_id
    else:
        params["location"] = f"{lat},{lon}"
    if cfg.heading is not None:
        params["heading"] = cfg.heading

    resp = requests.get(GOOGLE_STREET_VIEW_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def fetch_street_view_metadata(api_key: str, lat: float, lon: float) -> dict:
    params = {
        "location": f"{lat},{lon}",
        "key": api_key,
    }
    resp = requests.get(GOOGLE_STREET_VIEW_METADATA_ENDPOINT, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=600)
def places_nearby_count(api_key: str, lat: float, lon: float, radius_m: int, place_type: str) -> int:
    endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "type": place_type,
        "key": api_key,
    }
    total = 0
    next_token: Optional[str] = None
    attempts = 0
    while True:
        q = dict(params)
        if next_token:
            q.pop("location", None)
            q.pop("radius", None)
            q.pop("type", None)
            q["pagetoken"] = next_token
        r = requests.get(endpoint, params=q, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = str(data.get("status", "")).upper()
        if status == "OK":
            total += len(data.get("results", []))
            next_token = data.get("next_page_token")
            if next_token:
                time.sleep(2.1)
                continue
            return total
        if status in {"ZERO_RESULTS"}:
            return total
        if status in {"INVALID_REQUEST"} and attempts < 2 and data.get("next_page_token"):
            time.sleep(2.0)
            attempts += 1
            next_token = data.get("next_page_token")
            continue
        # For OVER_QUERY_LIMIT or REQUEST_DENIED, surface as zero with caching boundary
        return total


def compute_places_counts(api_key: str, lat: float, lon: float, radius_m: int, place_types: List[str]) -> List[dict]:
    results: List[dict] = []
    for t in place_types:
        try:
            c = places_nearby_count(api_key, lat, lon, radius_m, t)
        except Exception:
            c = 0
        results.append({"type": t, "count": int(c)})
    return results


@st.cache_resource(show_spinner=False)
def load_sam_model(model_type: str = "auto", checkpoint_path: Optional[str] = None):
    if checkpoint_path is None:
        # Try to get from env; if absent, raise with guidance.
        checkpoint_path = os.getenv("SAM_CHECKPOINT_PATH")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "SAM checkpoint not found. Select one from the models folder in the sidebar, upload a .pth, or set `SAM_CHECKPOINT_PATH`."
        )
    attempted_types: List[str] = []
    candidate_types: List[str]
    if model_type == "auto" or model_type not in ("vit_h", "vit_l", "vit_b"):
        inferred = infer_model_type_from_filename(checkpoint_path)
        candidate_types = [t for t in [inferred, "vit_h", "vit_l", "vit_b"] if t]
    else:
        candidate_types = [model_type]

    last_err: Optional[Exception] = None
    for cand in candidate_types:
        attempted_types.append(cand)
        try:
            sam = sam_model_registry[cand](checkpoint=checkpoint_path)
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam.to(device)
            except Exception:
                pass
            return SamAutomaticMaskGenerator(sam)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load SAM checkpoint with model types {attempted_types}. "
        "Ensure the checkpoint matches the selected model (e.g., sam_vit_h_*.pth -> vit_h, sam_vit_b_*.pth -> vit_b), and that it is SAM (not SAM2)."
    ) from last_err


def masks_to_overlay(image: np.ndarray, masks: List[dict], alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    for idx, m in enumerate(masks):
        seg = m.get("segmentation")
        if seg is None:
            continue
        color = np.random.default_rng(idx).integers(0, 255, size=3, dtype=np.uint8)
        colored = np.zeros_like(image, dtype=np.uint8)
        colored[seg] = color
        overlay = cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0)
    return overlay


def save_masks_as_pngs(masks: List[dict], base_dir: str) -> List[str]:
    os.makedirs(base_dir, exist_ok=True)
    saved = []
    for i, m in enumerate(masks):
        seg = m.get("segmentation")
        if seg is None:
            continue
        mask_img = (seg.astype(np.uint8) * 255)
        fp = os.path.join(base_dir, f"mask_{i:03d}.png")
        cv2.imwrite(fp, mask_img)
        saved.append(fp)
    return saved


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def main():
    st.set_page_config(page_title="PEDS - SAM Mask Generator", layout="wide")
    st.title("PEDS: Segment Anything on Street View")
    st.markdown(
        "Pick a point on the map, fetch Google Street View, run SAM auto masks, and download the masks."
    )

    with st.sidebar:
        st.header("Setup")
        google_api_key = st.text_input(
            "Google Maps Platform API Key",
            type="password",
            help="Required for Street View. Create in Google Cloud Console.",
            value=os.getenv("GOOGLE_MAPS_API_KEY", ""),
        )
        st.caption("Street View must be enabled for your key.")

        st.divider()
        st.header("SAM Model")
        model_type = st.selectbox(
            "Model Type",
            MODEL_TYPE_CHOICES,
            index=0,
            help="Choose the SAM backbone. 'auto' will infer from filename and try common types.",
        )

        models_dir = st.text_input(
            "Models directory",
            value=os.getenv("SAM_MODELS_DIR", "models"),
            help="Place your SAM .pth checkpoints here (e.g., sam_vit_h_4b8939.pth)",
        )
        available_ckpts = list_checkpoints_in_dir(models_dir)
        selected_ckpt = None
        if available_ckpts:
            selected_ckpt = st.selectbox(
                "Select checkpoint (.pth) from folder",
                available_ckpts,
                format_func=os.path.basename,
            )
            st.caption(f"Found {len(available_ckpts)} checkpoint file(s) in {os.path.abspath(os.path.expanduser(models_dir))}.")
        else:
            st.info("No .pth files found in the models directory.")

        uploaded_ckpt = st.file_uploader(
            "Optional: Upload SAM checkpoint (.pth)", type=["pth"], accept_multiple_files=False
        )
        ckpt_path = selected_ckpt
        if uploaded_ckpt is not None and not ckpt_path:
            tmp_dir = tempfile.mkdtemp()
            ckpt_path = os.path.join(tmp_dir, uploaded_ckpt.name)
            with open(ckpt_path, "wb") as f:
                f.write(uploaded_ckpt.read())
            st.success("Checkpoint uploaded.")

        st.divider()
        st.header("Street View Params")
        img_w = st.slider("Width", 256, 1280, 640, step=64)
        img_h = st.slider("Height", 256, 1280, 640, step=64)
        fov = st.slider("FOV", 30, 120, 90, step=5)
        pitch = st.slider("Pitch", -90, 90, 0, step=5)
        heading = st.slider("Heading", 0, 359, 0, step=1)
        auto_heading = st.checkbox("Let API choose heading", value=True)

        st.divider()
        run_btn = st.button("Fetch & Segment", type="primary")

        st.divider()
        st.header("Places Summary")
        enable_places = st.checkbox("Enable Places counts", value=True)
        place_types = st.multiselect("Place types", DEFAULT_PLACE_TYPES, default=DEFAULT_PLACE_TYPES[:5])
        places_radius = st.slider("Radius (meters)", 100, 2000, 400, step=50)
        places_btn = st.button("Compute Places summary")

    # Map selector
    st.subheader("Select Location")
    default_latlon = [37.773972, -122.431297]  # San Francisco
    m = folium.Map(location=default_latlon, zoom_start=13)
    marker = folium.Marker(location=default_latlon, draggable=True)
    marker.add_to(m)
    map_data = st_folium(m, width=800, height=500, returned_objects=["last_active_drawing", "last_object_clicked", "last_circle_polygon"])

    lat, lon = default_latlon
    # If user clicked on map, use that
    if map_data and map_data.get("last_object_clicked"):
        lat = map_data["last_object_clicked"]["lat"]
        lon = map_data["last_object_clicked"]["lng"]
    st.write(f"Lat: {lat:.6f}, Lon: {lon:.6f}")

    # Places summary panel
    if enable_places and places_btn:
        if not google_api_key:
            st.error("Please provide a Google Maps Platform API key to query Places.")
        else:
            with st.spinner("Querying Places API..."):
                summary = compute_places_counts(google_api_key, lat, lon, places_radius, place_types)
            st.subheader("Nearby Places summary")
            if summary:
                st.table(summary)
            else:
                st.info("No places found for the selected types and radius.")

    if run_btn:
        if not google_api_key:
            st.error("Please provide a Google Maps Platform API key.")
            st.stop()

        cfg = StreetViewConfig(
            size_w=img_w,
            size_h=img_h,
            fov=fov,
            pitch=pitch,
            heading=None if auto_heading else int(heading),
        )

        with st.spinner("Checking Street View availability..."):
            try:
                meta = fetch_street_view_metadata(google_api_key, lat, lon)
            except Exception as e:
                st.exception(e)
                st.stop()

        status = str(meta.get("status", "")).upper()
        if status != "OK":
            err = meta.get("error_message") or "Street View not available or key restriction issue."
            st.error(f"Street View metadata status: {status}. {err}")
            st.info(
                "Tips: Ensure billing is enabled; 'Street View Static API' is enabled; API key restrictions allow server-side requests (no HTTP referrer restriction for localhost), or use IP address restrictions; and the API is included in allowed APIs."
            )
            st.stop()

        pano_id = meta.get("pano_id")

        with st.spinner("Fetching Street View image..."):
            try:
                pil_img = fetch_street_view_image(google_api_key, lat, lon, cfg, pano_id=pano_id)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.image(pil_img, caption="Street View", use_container_width=True)

        with st.spinner("Loading SAM and generating masks (this may take a bit on first run)..."):
            try:
                mask_generator = load_sam_model(model_type=model_type, checkpoint_path=ckpt_path)
                image_cv = pil_to_cv(pil_img)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                masks = mask_generator.generate(image_rgb)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.success(f"Generated {len(masks)} masks")

        overlay = masks_to_overlay(image_cv, masks)
        st.image(cv_to_pil(overlay), caption="Mask Overlay", use_container_width=True)

        # Download section
        with st.expander("Download Masks"):
            tmp_out = tempfile.mkdtemp()
            paths = save_masks_as_pngs(masks, tmp_out)
            st.write(f"Saved {len(paths)} binary mask PNGs.")
            # Zip them
            import zipfile
            zip_path = os.path.join(tmp_out, "masks.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=os.path.basename(p))
            with open(zip_path, "rb") as f:
                st.download_button("Download masks.zip", data=f, file_name="masks.zip")


if __name__ == "__main__":
    main()



