## PEDS Walkability: Roadmap

### Vision
Automate PEDS scoring from public geospatial data and computer vision, enabling scalable, objective, and repeatable analysis across space and time.

### Phases
1. Prototype Hardening (current)
   - Improve error messaging, retries, metadata-first checks
   - Local models folder; auto-detect `vit_h/l/b`; consistent UI
   - Pin dependencies; config via `.env` and environment variables
   - Add on-disk cache for Street View metadata/image

2. Data Acquisition Services
   - Places Nearby counts with pagination and caching
   - Roads snap-to-roads; intersection heuristics; speed limits where permitted
   - Rate limiting, exponential backoff; cost/usage monitoring

3. Vision Pipeline
   - SAM masks with parameters; post-processing (merge/simplify)
   - Train object detectors for PEDS objects (signals, benches, curb cuts, obstructions)
   - Labeling plan; small gold dataset; evaluation harness (PR, mAP)

4. Feature Matrix Engineering
   - PEDS checklist → feature spec; schema in Parquet with provenance
   - Implement feature builders (vision + Places/Roads)
   - Unit tests, invariants, distribution checks

5. Ground Truth Dataset
   - Diverse sampling across cities; manual audits protocol
   - Inter-rater reliability (Cohen’s kappa) and rater training
   - Link segments to imagery and features

6. Predictive Model
   - Geo-stratified CV; leakage controls; hyperparameter search
   - Baselines (GBM/RF/linear) → calibrated model with uncertainty
   - Model card; SHAP/feature importance

7. Validation
   - External validation in held-out cities
   - Temporal robustness; imagery date sensitivity
   - Error analysis by land use, density, equity strata

8. Inference Service
   - REST API: lat/lon → feature builder → model → score
   - Batch pipeline for tiles/grids; caching; result store
   - Simple UI to export scores

9. MLOps/DevOps
   - DVC for data; MLflow for experiments/models
   - CI/CD: tests, lint, security; Docker/WSL2 images
   - Monitoring, drift detection, scheduled refresh

10. Compliance & Ethics
   - Google ToS compliance; storage limits for imagery
   - Privacy protections (blur or avoid persistence)
   - Bias checks and accessibility review

### Milestones & Acceptance
- M1: Hardened prototype (2 weeks)
  - App reliably fetches imagery, generates masks, basic caching, Places counts
- M2: Feature Matrix v1 (4–6 weeks)
  - 20+ validated PEDS features; tests and documentation
- M3: Ground Truth v1 (6–10 weeks)
  - 300–500 audited segments; reliability ≥ 0.7 on key items
- M4: Model v1 (8–12 weeks)
  - Meets target MAE; calibrated uncertainty; model card
- M5: Service v1 (12–16 weeks)
  - API + batch scoring; monitoring & logging

### Prioritized TODOs (next 1–2 sprints)
- Add disk cache for Street View metadata/images with TTL
- Implement Places Nearby counts summary in UI
- Draft PEDS→feature mapping doc and start 3 feature builders
- Set up MLflow local tracking; log runs
- Add unit tests for data clients and feature functions

### Risks & Mitigations
- API costs/limits → caching, backoff, budget caps
- ToS/privacy → minimize storage, blurring, legal review
- Label scarcity → start with few classes, active learning, transfer
- Geo bias → geo-stratified splits, external validation



