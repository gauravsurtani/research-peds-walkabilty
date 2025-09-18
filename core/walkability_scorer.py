"""
Walkability scoring system based on Places API data and density metrics
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class PlaceCategory:
    """Represents a category of places with counts and density"""
    name: str
    count: int
    density_per_km: float
    rating: str  # 'excellent', 'good', 'fair', 'poor'
    score: float  # 0.0 to 1.0
    weight: float  # Category weight in overall score

@dataclass
class WalkabilityScore:
    """Complete walkability analysis results"""
    overall_score: float  # 0-100
    category_scores: Dict[str, PlaceCategory]
    total_places: int
    analysis_radius_m: int
    
    def get_score_breakdown(self) -> Dict[str, float]:
        """Get weighted contribution of each category to overall score"""
        return {
            category.name: category.score * category.weight * 100
            for category in self.category_scores.values()
        }

class WalkabilityScorer:
    """Calculates walkability scores based on Places API data"""
    
    def __init__(self):
        # Scoring weights from your archive (healthcare prioritized)
        self.weights = {
            "healthcare": 0.30,
            "transit": 0.20,
            "amenities": 0.15,
            "retail": 0.10,
            "restaurants": 0.10,
            "services": 0.10,
            "entertainment": 0.05
        }
        
        # Place type mappings from your archive
        self.place_categories = {
            "healthcare": [
                "hospital", "doctor", "dentist", "pharmacy", "clinic",
                "physiotherapist", "veterinary_care", "health"
            ],
            "transit": [
                "bus_station", "subway_station", "train_station", 
                "transit_station", "light_rail_station"
            ],
            "amenities": [
                "bank", "post_office", "library", "school", "university",
                "parking", "atm", "government_office"
            ],
            "retail": [
                "convenience_store", "department_store", "clothing_store", 
                "shoe_store", "jewelry_store", "book_store", "supermarket",
                "grocery_or_supermarket", "shopping_mall"
            ],
            "restaurants": [
                "restaurant", "cafe", "bakery", "food", "meal_takeaway",
                "bar", "night_club"
            ],
            "services": [
                "beauty_salon", "barber_shop", "laundry", "dry_cleaner",
                "car_repair", "gas_station", "locksmith", "real_estate_agency"
            ],
            "entertainment": [
                "movie_theater", "museum", "art_gallery", "park", "gym",
                "amusement_park", "zoo", "aquarium", "bowling_alley"
            ]
        }
        
        # Density thresholds from your archive (per km)
        self.thresholds = {
            "healthcare": {
                'excellent': 3.0,
                'good': 2.0,
                'fair': 1.0,
                'poor': 0.0
            },
            "transit": {
                'excellent': 2.0,
                'good': 1.0,
                'fair': 0.5,
                'poor': 0.0
            },
            "amenities": {
                'excellent': 5.0,
                'good': 3.0,
                'fair': 1.0,
                'poor': 0.0
            },
            "retail": {
                'excellent': 8.0,
                'good': 5.0,
                'fair': 2.0,
                'poor': 0.0
            },
            "restaurants": {
                'excellent': 6.0,
                'good': 4.0,
                'fair': 2.0,
                'poor': 0.0
            },
            "services": {
                'excellent': 4.0,
                'good': 2.0,
                'fair': 1.0,
                'poor': 0.0
            },
            "entertainment": {
                'excellent': 3.0,
                'good': 2.0,
                'fair': 1.0,
                'poor': 0.0
            }
        }
        
        # Score values for ratings
        self.rating_scores = {
            'excellent': 1.0,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.2
        }
    
    def categorize_places(self, places_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize places from Places API results into walkability categories
        
        Args:
            places_data: List of place dictionaries from Places API
            
        Returns:
            Dictionary mapping category names to lists of places
        """
        categorized = {category: [] for category in self.place_categories.keys()}
        
        for place in places_data:
            place_types = place.get('types', [])
            
            # Check which category this place belongs to
            for category, category_types in self.place_categories.items():
                if any(place_type in category_types for place_type in place_types):
                    categorized[category].append(place)
                    break  # Place goes in first matching category only
        
        return categorized
    
    def calculate_density_per_km(self, count: int, radius_m: int) -> float:
        """
        Calculate density per square kilometer
        
        Args:
            count: Number of places found
            radius_m: Search radius in meters
            
        Returns:
            Density per square kilometer
        """
        if radius_m <= 0:
            return 0.0
        
        # Calculate area in square kilometers
        area_km2 = math.pi * (radius_m / 1000) ** 2
        
        return count / area_km2 if area_km2 > 0 else 0.0
    
    def get_density_rating(self, density: float, category: str) -> str:
        """
        Get rating (excellent/good/fair/poor) based on density thresholds
        
        Args:
            density: Density per square kilometer
            category: Category name
            
        Returns:
            Rating string
        """
        thresholds = self.thresholds.get(category, self.thresholds['amenities'])
        
        if density >= thresholds['excellent']:
            return 'excellent'
        elif density >= thresholds['good']:
            return 'good'
        elif density >= thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def calculate_walkability_score(self, places_data: List[Dict], radius_m: int) -> WalkabilityScore:
        """
        Calculate comprehensive walkability score from Places API data
        
        Args:
            places_data: List of places from Places API
            radius_m: Search radius in meters
            
        Returns:
            WalkabilityScore with detailed breakdown
        """
        # Categorize places
        categorized_places = self.categorize_places(places_data)
        
        # Calculate scores for each category
        category_scores = {}
        overall_weighted_score = 0.0
        
        for category, places in categorized_places.items():
            count = len(places)
            density = self.calculate_density_per_km(count, radius_m)
            rating = self.get_density_rating(density, category)
            score = self.rating_scores[rating]
            weight = self.weights[category]
            
            category_scores[category] = PlaceCategory(
                name=category,
                count=count,
                density_per_km=density,
                rating=rating,
                score=score,
                weight=weight
            )
            
            # Add weighted contribution to overall score
            overall_weighted_score += score * weight
        
        # Convert to 0-100 scale
        overall_score = overall_weighted_score * 100
        
        return WalkabilityScore(
            overall_score=overall_score,
            category_scores=category_scores,
            total_places=len(places_data),
            analysis_radius_m=radius_m
        )
    
    def get_recommendations(self, walkability_score: WalkabilityScore) -> List[str]:
        """
        Generate recommendations based on walkability analysis
        
        Args:
            walkability_score: Calculated walkability score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check each category for improvement opportunities
        for category, place_category in walkability_score.category_scores.items():
            if place_category.rating == 'poor':
                if category == 'healthcare':
                    recommendations.append(f"⚕️ Healthcare access is limited - consider proximity to medical facilities")
                elif category == 'transit':
                    recommendations.append(f"🚌 Public transit access is poor - check bus/train connections")
                elif category == 'retail':
                    recommendations.append(f"🛒 Shopping options are limited - few retail stores nearby")
                elif category == 'restaurants':
                    recommendations.append(f"🍽️ Dining options are limited - few restaurants/cafes nearby")
        
        # Overall score recommendations
        if walkability_score.overall_score >= 80:
            recommendations.insert(0, "🌟 Excellent walkability! This area has great pedestrian amenities.")
        elif walkability_score.overall_score >= 60:
            recommendations.insert(0, "👍 Good walkability with room for improvement in some areas.")
        elif walkability_score.overall_score >= 40:
            recommendations.insert(0, "⚠️ Fair walkability - some key amenities may be missing.")
        else:
            recommendations.insert(0, "🚨 Poor walkability - limited pedestrian amenities nearby.")
        
        return recommendations