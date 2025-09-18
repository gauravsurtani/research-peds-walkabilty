# PEDS Walkability: Street View + Segment Anything Demo

This app lets you pick a latitude/longitude on a map, fetch the Google Street View image for that location, run Meta's Segment Anything Model (SAM) automatic mask generator, visualize overlays, and download binary masks.

## Prerequisites

- Python 3.10+
- A Google Maps Platform API key with Street View Static API enabled
- A SAM checkpoint file (e.g., `sam_vit_h_4b8939.pth`). Place it in a local `models` folder (default) or set the `SAM_MODELS_DIR` env var. You can still upload in the UI if preferred.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
set GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE
set SAM_MODELS_DIR=C:\\ai_projects\\parking\\models  # optional; defaults to ./models
streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit. Click on the map to pick a point, adjust Street View parameters in the sidebar, and click "Fetch & Segment".

## Project docs

- Flow: see `FLOW.md` for the end-to-end architecture and module responsibilities.
- Roadmap: see `ROADMAP.md` for phases, milestones, and prioritized TODOs.
- Spec: see `.kiro/specs/peds-walkability-merge/` for comprehensive requirements, design, and implementation plan.

## Setup links (Google Maps Platform)

- Create/select project: https://console.cloud.google.com/cloud-resource-manager
- Enable billing: https://console.cloud.google.com/billing
- Create API key (Credentials): https://console.cloud.google.com/apis/credentials
- Manage quotas: https://console.cloud.google.com/iam-admin/quotas
- API key security best practices: https://developers.google.com/maps/api-security-best-practices

Enable these APIs (open link and click Enable):
- Street View Static API: https://console.cloud.google.com/apis/library/street-view-image-backend.googleapis.com
- Places API: https://console.cloud.google.com/apis/library/places-backend.googleapis.com
- Roads API: https://console.cloud.google.com/apis/library/roads.googleapis.com
- Directions API: https://console.cloud.google.com/apis/library/directions-backend.googleapis.com

Official docs and billing pages:
- Street View Static API overview: https://developers.google.com/maps/documentation/streetview/images
- Street View usage & billing: https://developers.google.com/maps/documentation/streetview/usage-and-billing
- Places API overview: https://developers.google.com/maps/documentation/places/web-service/overview
- Places usage & billing: https://developers.google.com/maps/documentation/places/web-service/usage-and-billing
- Roads API overview: https://developers.google.com/maps/documentation/roads/intro
- Roads usage & billing: https://developers.google.com/maps/documentation/roads/usage-and-billing
- Directions API overview: https://developers.google.com/maps/documentation/directions/overview
- Directions usage & billing: https://developers.google.com/maps/documentation/directions/usage-and-billing
- Get started hub: https://developers.google.com/maps/get-started

## Outputs

- Mask overlay visualization
- Downloadable `masks.zip` containing per-object binary masks (`mask_###.png`)

## Notes

- If you have a CUDA GPU and drivers installed, SAM will use it automatically