# Reference implementation from PEDS-archive
# This file is kept for reference during implementation of task 6.1
# Original path: PEDS-archive/app/scoring.py

# Key scoring thresholds to implement:
SCORING_THRESHOLDS = {
    "TRANSIT": {
        'excellent': 2.0,  # 2+ transit stops per km
        'good': 1.0,       # 1+ transit stops per km
        'fair': 0.5,       # 0.5+ transit stops per km
        'poor': 0.0
    },
    "HEALTHCARE": {
        'excellent': 3.0,  # 3+ healthcare facilities per km
        'good': 2.0,       # 2+ healthcare facilities per km
        'fair': 1.0,       # 1+ healthcare facilities per km
        'poor': 0.0
    },
    "AMENITY": {
        'excellent': 5.0,  # 5+ amenities per km
        'good': 3.0,       # 3+ amenities per km
        'fair': 1.0,       # 1+ amenities per km
        'poor': 0.0
    },
    "RETAIL": {
        'excellent': 8.0,  # 8+ retail locations per km
        'good': 5.0,       # 5+ retail locations per km
        'fair': 2.0,       # 2+ retail locations per km
        'poor': 0.0
    }
}

# Score mapping:
# excellent = 1.0, good = 0.8, fair = 0.6, poor = 0.2