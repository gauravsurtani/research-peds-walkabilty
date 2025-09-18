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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Segment Anything
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Import our walkability scorer and SAM street analyzer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.walkability_scorer import WalkabilityScorer
from core.sam_street_analyzer import SAMStreetAnalyzer


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


@st.cache_data(show_spinner=False, ttl=600)
def get_all_places_data(api_key: str, lat: float, lon: float, radius_m: int) -> List[dict]:
    """
    Get detailed place data for walkability analysis (not just counts)
    """
    endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "key": api_key,
    }
    
    all_places = []
    next_token: Optional[str] = None
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        q = dict(params)
        if next_token:
            q.pop("location", None)
            q.pop("radius", None)
            q["pagetoken"] = next_token
        
        try:
            r = requests.get(endpoint, params=q, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            status = str(data.get("status", "")).upper()
            if status == "OK":
                places = data.get("results", [])
                all_places.extend(places)
                
                next_token = data.get("next_page_token")
                if next_token:
                    time.sleep(2.1)  # Required delay for next page token
                    attempts += 1
                    continue
                else:
                    break
            elif status in {"ZERO_RESULTS"}:
                break
            else:
                # For other errors, return what we have
                break
                
        except Exception as e:
            st.warning(f"Error fetching places data: {str(e)}")
            break
    
    return all_places


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
    """Create a colorful overlay of SAM masks on the original image"""
    if len(masks) == 0:
        return image
    
    overlay = image.copy()
    
    # Sort masks by area (largest first) for better visualization
    sorted_masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
    
    # Use distinct colors for better visibility
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green  
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 192, 203], # Pink
        [0, 128, 0],    # Dark Green
    ]
    
    for idx, mask in enumerate(sorted_masks):
        seg = mask.get("segmentation")
        if seg is None:
            continue
            
        # Use predefined colors, cycling through them
        color = colors[idx % len(colors)]
        
        # Create colored mask
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[seg] = color
        
        # Blend with overlay
        mask_area = seg.astype(np.uint8) * 255
        mask_area = np.stack([mask_area] * 3, axis=-1)
        
        # Apply alpha blending only where mask exists
        overlay = np.where(mask_area > 0, 
                          cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0),
                          overlay)
    
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
    
    # Initialize component state management
    if 'component_reset_counter' not in st.session_state:
        st.session_state.component_reset_counter = 0

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
    
    # Initialize session state for coordinates
    if 'lat' not in st.session_state:
        st.session_state.lat = 37.773972
    if 'lon' not in st.session_state:
        st.session_state.lon = -122.431297
    
    # Initialize preset selection state
    if 'selected_preset' not in st.session_state:
        st.session_state.selected_preset = "Custom Location"
    
    # Preset locations for easy testing
    preset_locations = {
        "San Francisco - Market Street": (37.7749, -122.4194),
        "New York - Times Square": (40.7580, -73.9855),
        "London - Big Ben": (51.4994, -0.1245),
        "Paris - Eiffel Tower": (48.8584, 2.2945),
        "Tokyo - Shibuya Crossing": (35.6598, 139.7006),
        "Custom Location": "Use coordinates below or click on map"
    }
    
    # Check if current coordinates match any preset (with tolerance)
    current_preset = "Custom Location"
    for name, coords in preset_locations.items():
        if name != "Custom Location" and isinstance(coords, tuple):
            lat_diff = abs(coords[0] - st.session_state.lat)
            lon_diff = abs(coords[1] - st.session_state.lon)
            if lat_diff < 0.001 and lon_diff < 0.001:  # Close enough to preset
                current_preset = name
                break
    
    selected_preset = st.selectbox(
        "Choose a preset location or select 'Custom Location' to use coordinates below:",
        list(preset_locations.keys()),
        index=list(preset_locations.keys()).index(current_preset)
    )
    
    # Only update coordinates if user actively selected a different preset
    if selected_preset != st.session_state.selected_preset and selected_preset != "Custom Location":
        st.session_state.lat, st.session_state.lon = preset_locations[selected_preset]
        st.session_state.map_update_counter += 1
        st.session_state.selected_preset = selected_preset
        st.rerun()
    elif selected_preset == "Custom Location":
        st.session_state.selected_preset = "Custom Location"
    
    # Manual coordinate input
    col1, col2 = st.columns(2)
    with col1:
        manual_lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f", step=0.000001)
    with col2:
        manual_lon = st.number_input("Longitude", value=st.session_state.lon, format="%.6f", step=0.000001)
    
    # Update session state if manual input changed
    if manual_lat != st.session_state.lat or manual_lon != st.session_state.lon:
        st.session_state.lat = manual_lat
        st.session_state.lon = manual_lon
    
    # Initialize map click tracking in session state
    if 'last_processed_click' not in st.session_state:
        st.session_state.last_processed_click = None
    
    # Initialize map update counter to force re-render when coordinates change
    if 'map_update_counter' not in st.session_state:
        st.session_state.map_update_counter = 0
    
    # Create map with click handler
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=15)
    
    # Add click handler to map
    m.add_child(folium.LatLngPopup())
    
    # Add a marker at current location
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"📍 Selected Location<br/>Lat: {st.session_state.lat:.6f}<br/>Lon: {st.session_state.lon:.6f}",
        tooltip="Current Location - Click anywhere on map to change",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Instructions with troubleshooting
    st.markdown("🗺️ **Click anywhere on the map to select a new location**")
    st.markdown("💡 *If map clicking doesn't work, use the coordinate inputs above or quick location buttons below*")
    
    # Display map with dynamic key that changes when coordinates change
    map_key = f"main_map_{st.session_state.map_update_counter}_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}"
    map_data = st_folium(
        m, 
        width=800, 
        height=400,
        returned_objects=["last_clicked"],  # Simplified to reduce component messages
        key=map_key,  # Dynamic key to force update when coordinates change
        feature_group_to_add=None,
        zoom=15
    )
    
    # Show debug info toggle
    show_debug = st.checkbox("🔍 Show map debug info", value=False)
    if show_debug:
        st.json(map_data)
    
    # Handle map clicks with improved state management
    if map_data:
        clicked_coords = None
        
        # Try to get click coordinates (simplified)
        if map_data.get("last_clicked"):
            clicked_coords = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        
        # Process the click if we have new coordinates
        if clicked_coords and clicked_coords != st.session_state.last_processed_click:
            new_lat, new_lon = clicked_coords
            
            # Check if this is a significant change from current location
            lat_diff = abs(new_lat - st.session_state.lat)
            lon_diff = abs(new_lon - st.session_state.lon)
            
            if lat_diff > 0.0001 or lon_diff > 0.0001:
                # Update the location and force map refresh
                st.session_state.lat = new_lat
                st.session_state.lon = new_lon
                st.session_state.last_processed_click = clicked_coords
                st.session_state.map_update_counter += 1  # Force map re-render
                st.session_state.selected_preset = "Custom Location"  # Set to custom when clicking map
                
                # Show success message
                st.success(f"📍 Location updated! New coordinates: {new_lat:.6f}, {new_lon:.6f}")
                st.rerun()  # Immediately refresh to show new marker position
    
    # Display current coordinates with additional controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📍 Current Location: {st.session_state.lat:.6f}, {st.session_state.lon:.6f}")
    with col2:
        if st.button("🎯 Refresh Map", help="Refresh map to show current location"):
            # Force map update and clear click tracking
            st.session_state.last_processed_click = None
            st.session_state.map_update_counter += 1
            st.rerun()
    
    # Alternative: Quick location buttons for testing
    st.markdown("**🚀 Quick Test Locations:**")
    test_cols = st.columns(4)
    
    test_locations = [
        ("🌉 SF Market St", 37.7749, -122.4194),
        ("🗽 NYC Times Sq", 40.7580, -73.9855), 
        ("🏛️ London Big Ben", 51.4994, -0.1245),
        ("🗼 Paris Eiffel", 48.8584, 2.2945)
    ]
    
    for i, (name, lat_test, lon_test) in enumerate(test_locations):
        with test_cols[i]:
            if st.button(name, key=f"test_loc_{i}"):
                st.session_state.lat = lat_test
                st.session_state.lon = lon_test
                st.session_state.last_processed_click = None  # Reset click tracking
                st.session_state.map_update_counter += 1  # Force map update
                st.session_state.selected_preset = "Custom Location"  # Set to custom since it's a quick jump
                st.success(f"Jumped to {name}!")
                st.rerun()
    
    # Additional coordinate picker
    st.markdown("**🎯 Or try these interesting locations:**")
    interesting_cols = st.columns(3)
    
    interesting_locations = [
        ("🏖️ Santa Monica Pier", 34.0089, -118.4973),
        ("🌁 Golden Gate Bridge", 37.8199, -122.4783),
        ("🏙️ Manhattan Bridge", 40.7061, -73.9969)
    ]
    
    for i, (name, lat_test, lon_test) in enumerate(interesting_locations):
        with interesting_cols[i]:
            if st.button(name, key=f"interesting_loc_{i}"):
                st.session_state.lat = lat_test
                st.session_state.lon = lon_test
                st.session_state.last_processed_click = None  # Reset click tracking
                st.session_state.map_update_counter += 1  # Force map update
                st.session_state.selected_preset = "Custom Location"  # Set to custom since it's a quick jump
                st.success(f"Jumped to {name}!")
                st.rerun()
    
    # Places summary panel
    if enable_places and places_btn:
        if not google_api_key:
            st.error("Please provide a Google Maps Platform API key to query Places.")
        else:
            # Get current coordinates for places query
            lat, lon = st.session_state.lat, st.session_state.lon
            with st.spinner("Querying Places API and calculating walkability score..."):
                # Get all places data (not just counts)
                all_places_data = get_all_places_data(google_api_key, lat, lon, places_radius)
                
                # Calculate walkability score using real features
                walkability_scorer = WalkabilityScorer()
                walkability_score = walkability_scorer.calculate_walkability_score(all_places_data, places_radius)
                
            st.subheader("🚶 Walkability Analysis")
            
            # Display overall score prominently
            score_color = "green" if walkability_score.overall_score > 70 else "orange" if walkability_score.overall_score > 40 else "red"
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### Overall Walkability Score")
                st.markdown(f"<h1 style='color: {score_color}; text-align: center'>{walkability_score.overall_score:.1f}/100</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center'>Based on {walkability_score.total_places} nearby places</p>", unsafe_allow_html=True)
            
            with col2:
                # Category breakdown
                st.markdown("### Category Breakdown")
                breakdown = walkability_score.get_score_breakdown()
                
                for category, place_category in walkability_score.category_scores.items():
                    # Create rating emoji
                    rating_emoji = {
                        'excellent': '🌟',
                        'good': '👍', 
                        'fair': '⚠️',
                        'poor': '🚨'
                    }
                    
                    emoji = rating_emoji.get(place_category.rating, '❓')
                    contribution = breakdown[category]
                    
                    st.write(f"{emoji} **{category.title()}**: {place_category.count} places "
                            f"({place_category.density_per_km:.1f}/km²) - {place_category.rating} "
                            f"(+{contribution:.1f} points)")
            
            # Recommendations
            recommendations = walkability_scorer.get_recommendations(walkability_score)
            if recommendations:
                st.subheader("💡 Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # Detailed breakdown table
            with st.expander("📊 Detailed Category Analysis"):
                category_data = []
                for category, place_category in walkability_score.category_scores.items():
                    category_data.append({
                        "Category": category.title(),
                        "Count": place_category.count,
                        "Density/km²": f"{place_category.density_per_km:.2f}",
                        "Rating": place_category.rating.title(),
                        "Score": f"{place_category.score:.2f}",
                        "Weight": f"{place_category.weight:.0%}",
                        "Contribution": f"{breakdown[category]:.1f}"
                    })
                st.table(category_data)

    if run_btn:
        if not google_api_key:
            st.error("Please provide a Google Maps Platform API key.")
            st.stop()

        # Get the current coordinates from session state right before processing
        lat, lon = st.session_state.lat, st.session_state.lon
        
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
                
                # Debug info
                st.write(f"✅ Successfully fetched Street View image")
                st.write(f"📐 Image dimensions: {pil_img.size[0]} x {pil_img.size[1]}")
                st.write(f"🎨 Image mode: {pil_img.mode}")
                
            except Exception as e:
                st.error(f"❌ Failed to fetch Street View image: {str(e)}")
                st.exception(e)
                st.stop()

        # Display original Street View image
        st.subheader("📸 Street View Image")
        st.image(pil_img, caption=f"Street View at {lat:.4f}, {lon:.4f}", use_container_width=True)
        
        # Check image properties
        st.write(f"Image size: {pil_img.size[0]} x {pil_img.size[1]} pixels")

        with st.spinner("🤖 Loading SAM and generating masks (this may take a moment on first run)..."):
            try:
                mask_generator = load_sam_model(model_type=model_type, checkpoint_path=ckpt_path)
                
                # Convert PIL to numpy array for SAM
                image_array = np.array(pil_img)
                
                # Generate masks
                masks = mask_generator.generate(image_array)
                
            except Exception as e:
                st.error(f"SAM processing failed: {str(e)}")
                st.exception(e)
                st.stop()

        st.success(f"✅ Generated {len(masks)} masks")

        if len(masks) > 0:
            # Try to do more realistic SAM analysis using position heuristics
            with st.spinner("🔍 Analyzing street view features using position heuristics..."):
                try:
                    # Initialize SAM street analyzer
                    sam_analyzer = SAMStreetAnalyzer()
                    sam_analyzer.load_model(ckpt_path, model_type)
                    
                    # Analyze the street view image (now returns both analysis and masks)
                    street_analysis, coverage_masks = sam_analyzer.analyze_street_view(np.array(pil_img))
                    
                    # Create colored segmentation overlay
                    analysis_overlay = sam_analyzer.create_segmentation_overlay(
                        np.array(pil_img), coverage_masks, alpha=0.6
                    )
                    
                except Exception as e:
                    st.warning(f"Could not perform detailed SAM analysis: {str(e)}")
                    street_analysis = None
                    analysis_overlay = None
                    coverage_masks = None
            
            # Create generic overlay
            generic_overlay = masks_to_overlay(np.array(pil_img), masks, alpha=0.5)
            
            # Display results
            if street_analysis:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🎭 Generic SAM Masks")
                    st.image(generic_overlay, caption="All SAM Masks", use_container_width=True)
                
                with col2:
                    st.subheader("🏙️ Street Segmentation")
                    st.image(analysis_overlay, caption="Buildings (Red), Footpaths (Green), Roads (Gray), etc.", use_container_width=True)
                    st.caption("🔍 Colors: Road=Gray, Footpath=Green, Buildings=Red, Trees=Dark Green, Sky=Blue, Vehicles=Blue, Signs=Yellow")
                
                with col3:
                    st.subheader("📊 Visual Features")
                    
                    # Coverage analysis
                    st.write("**Surface Coverage:**")
                    st.write(f"• Road: {street_analysis.road_coverage:.1f}%")
                    st.write(f"• Footpath: {street_analysis.footpath_coverage:.1f}%")
                    st.write(f"• Buildings: {street_analysis.building_coverage:.1f}%")
                    st.write(f"• Vegetation: {street_analysis.vegetation_coverage:.1f}%")
                    st.write(f"• Sky: {street_analysis.sky_coverage:.1f}%")
                    
                    st.write("**Objects Detected:**")
                    st.write(f"• Vehicles: {street_analysis.vehicle_count}")
                    st.write(f"• Signs: {street_analysis.sign_count}")
                    st.write(f"• Total segments: {street_analysis.total_segments}")
                    
                    # Walkability indicators
                    indicators = street_analysis.get_walkability_indicators()
                    st.write("**Walkability Indicators:**")
                    for indicator, value in indicators.items():
                        if isinstance(value, (int, float)):
                            if indicator in ['footpath_availability', 'green_space']:
                                emoji = "✅" if value > 10 else "⚠️" if value > 5 else "❌"
                            elif indicator in ['road_dominance', 'traffic_density']:
                                emoji = "❌" if value > 50 else "⚠️" if value > 25 else "✅"
                            else:
                                emoji = "ℹ️"
                            
                            if isinstance(value, float):
                                st.write(f"• {indicator.replace('_', ' ').title()}: {value:.1f}% {emoji}")
                            else:
                                st.write(f"• {indicator.replace('_', ' ').title()}: {value} {emoji}")
                    
                    st.info("ℹ️ Analysis uses position heuristics. Results are approximate.")
            
            else:
                # Fallback to simple display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎭 SAM Segmentation Masks")
                    st.image(generic_overlay, caption="SAM Mask Overlay", use_container_width=True)
                
                with col2:
                    st.subheader("📊 Basic Mask Statistics")
                    st.write(f"**Total masks generated:** {len(masks)}")
                    
                    # Show mask size distribution
                    mask_areas = [mask.get('area', 0) for mask in masks]
                    if mask_areas:
                        st.write(f"**Largest mask area:** {max(mask_areas):,} pixels")
                        st.write(f"**Smallest mask area:** {min(mask_areas):,} pixels")
                        st.write(f"**Average mask area:** {sum(mask_areas)//len(mask_areas):,} pixels")
                    
                    st.info("ℹ️ For detailed analysis, ensure SAM model is properly loaded.")
        
        else:
            st.warning("⚠️ No masks were generated. Try a different location or check your SAM model.")

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



