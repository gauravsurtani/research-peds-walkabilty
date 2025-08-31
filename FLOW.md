## PEDS Walkability: System Flow

### Purpose
High-level flow of the prototype and the full pipeline from input coordinates to a predicted PEDS score.

### End-to-end flow
```mermaid
graph TD
    A[User selects lat/lon] --> B[Street View Metadata]
    B -->|status OK + pano_id| C[Street View Image Fetch]
    B -->|status not OK| B1[Guidance: enable API, fix key restrictions]
    C --> D[SAM Automatic Mask Generation]
    D --> E[Overlay Visualization]
    D --> F[Binary Masks (.png)]
    F --> G[Feature Extraction (vision)]
    C --> H[Context Features (Places/Roads/Directions)]
    G --> I[Feature Matrix]
    H --> I
    I --> J[Inference: PEDS Prediction Model]
    J --> K[Predicted PEDS Score + Uncertainty]
```

### Modules and responsibilities
- Street View Client: metadata-first checks; fetch images; on-disk cache with TTL
- SAM Masking: load local checkpoint from `models/`; generate segmentation masks
- Feature Extractors:
  - Vision-derived: obstructions, curb cuts, signals, benches, sidewalk width proxies
  - Places: POI densities by type within buffers
  - Roads/Directions: snapped paths, intersection counts, speeds (where permitted)
- Feature Matrix: standardized schema (Parquet); provenance (pano_id, date)
- Model: supervised regressor to predict PEDS; calibrated uncertainty
- UI/API: Streamlit UI for demo; REST API for batch/online scoring

### Configuration
- Environment variables
  - `GOOGLE_MAPS_API_KEY`: required for Street View/Places/Roads
  - `SAM_MODELS_DIR`: folder with SAM checkpoints (default: `./models`)
  - Optional future: `CACHE_DIR`, `REQUESTS_PER_MIN`, `PLACES_TYPES`
- Local folders
  - `models/` — SAM `.pth` checkpoints
  - `data/cache/` — cached metadata/images (planned)

### Error handling (key examples)
- Street View metadata `status != OK`: show status and remediation (enable API, billing, key restrictions)
- HTTP 403/429/5xx: retries with backoff; surface actionable messages
- SAM checkpoint mismatch: infer model type; clear guidance to use matching `vit_h/l/b` checkpoints

### Future extensions
- Automatic heading sweep to improve coverage
- Queue-based batch scoring by tiles/geohash
- Privacy filters if persisting images (blur faces/plates)

### Current tasks
- Segmentation smoke test
  - Load SAM checkpoint from `models/`
  - Select a lat/lon with known Street View coverage
  - Verify Street View metadata returns `status=OK`
  - Fetch image and run SAM automatic mask generator
  - Confirm overlay renders and `masks.zip` downloads with non-empty PNGs
  - Record pass/fail and any errors (403, model mismatch, GPU out-of-memory)
- Key setup validation
  - Ensure billing enabled and “Street View Static API” turned on
  - Relax application restrictions (None or IP allowlist), and restrict to necessary APIs
  - Re-test metadata endpoint if 403

