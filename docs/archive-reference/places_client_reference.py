# Reference implementation from PEDS-archive
# This file is kept for reference during implementation of task 4.1
# Original path: PEDS-archive/app/places_client.py

# Key features to implement:
# - Healthcare-prioritized categorization (30% weight)
# - Rate limiting with configurable requests per minute
# - Comprehensive place type mapping
# - Distance calculations for street segments
# - API usage tracking and statistics

# Scoring weights from archive config:
WALKABILITY_WEIGHTS = {
    "healthcare_density": 0.30,
    "transit_access": 0.20,
    "amenity_density": 0.15,
    "retail_density": 0.10,
    "restaurant_density": 0.10,
    "service_density": 0.10,
    "entertainment_density": 0.05
}

# Place categories from archive:
PLACE_CATEGORIES = {
    "healthcare": ["hospital", "doctor", "dentist", "pharmacy"],
    "retail": ["convenience_store", "department_store", "clothing_store", "shoe_store", "jewelry_store", "book_store"],
    "restaurants": ["restaurant", "cafe", "bakery", "food", "meal_takeaway"],
    "amenities": ["bus_station", "subway_station", "train_station", "parking", "bank", "post_office", "library", "school"],
    "services": ["beauty_salon", "barber_shop", "laundry", "dry_cleaner", "car_repair", "gas_station", "atm"],
    "entertainment": ["movie_theater", "museum", "art_gallery", "park", "gym"]
}