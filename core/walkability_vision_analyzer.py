"""
Walkability-focused computer vision analysis using SAM masks
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import cv2

@dataclass
class WalkabilityFeatures:
    """Visual features relevant to pedestrian walkability"""
    sidewalk_coverage: float  # Percentage of image with sidewalks
    crosswalk_detected: bool  # Presence of crosswalks
    street_furniture_count: int  # Benches, bus stops, etc.
    vegetation_coverage: float  # Trees, green spaces
    lighting_elements: int  # Street lights, signs
    obstacles_detected: int  # Barriers, construction
    overall_walkability_score: float  # 0-100 visual walkability score

@dataclass 
class MaskAnalysis:
    """Analysis of individual SAM mask"""
    mask_id: int
    area_pixels: int
    area_percentage: float
    predicted_category: str
    confidence: float
    walkability_relevance: str  # 'positive', 'negative', 'neutral'

class WalkabilityVisionAnalyzer:
    """Analyzes SAM masks for pedestrian walkability features"""
    
    def __init__(self):
        # Define walkability-relevant categories and their impact
        self.walkability_categories = {
            # Positive features (improve walkability)
            'sidewalk': {'impact': 'positive', 'weight': 0.3},
            'crosswalk': {'impact': 'positive', 'weight': 0.25},
            'bench': {'impact': 'positive', 'weight': 0.1},
            'bus_stop': {'impact': 'positive', 'weight': 0.15},
            'street_light': {'impact': 'positive', 'weight': 0.1},
            'tree': {'impact': 'positive', 'weight': 0.08},
            'bike_rack': {'impact': 'positive', 'weight': 0.05},
            'sign': {'impact': 'positive', 'weight': 0.03},
            
            # Negative features (hinder walkability)
            'construction': {'impact': 'negative', 'weight': -0.2},
            'barrier': {'impact': 'negative', 'weight': -0.15},
            'parked_car_blocking': {'impact': 'negative', 'weight': -0.1},
            'debris': {'impact': 'negative', 'weight': -0.05},
            
            # Neutral features
            'building': {'impact': 'neutral', 'weight': 0.0},
            'road': {'impact': 'neutral', 'weight': 0.0},
            'sky': {'impact': 'neutral', 'weight': 0.0},
        }
    
    def analyze_masks_for_walkability(self, image: np.ndarray, masks: List[Dict]) -> WalkabilityFeatures:
        """
        Analyze SAM masks to extract walkability-relevant features
        
        Args:
            image: Original street view image
            masks: List of SAM-generated masks
            
        Returns:
            WalkabilityFeatures with extracted visual metrics
        """
        if not masks:
            return WalkabilityFeatures(
                sidewalk_coverage=0.0,
                crosswalk_detected=False,
                street_furniture_count=0,
                vegetation_coverage=0.0,
                lighting_elements=0,
                obstacles_detected=0,
                overall_walkability_score=0.0
            )
        
        # Analyze each mask
        mask_analyses = []
        for i, mask in enumerate(masks):
            analysis = self._analyze_individual_mask(image, mask, i)
            mask_analyses.append(analysis)
        
        # Extract walkability features
        return self._extract_walkability_features(mask_analyses, image.shape)
    
    def _analyze_individual_mask(self, image: np.ndarray, mask: Dict, mask_id: int) -> MaskAnalysis:
        """Analyze individual mask for walkability relevance"""
        segmentation = mask.get('segmentation', np.zeros_like(image[:,:,0]))
        area_pixels = int(np.sum(segmentation))
        total_pixels = image.shape[0] * image.shape[1]
        area_percentage = (area_pixels / total_pixels) * 100
        
        # Simple heuristic-based categorization (could be replaced with trained classifier)
        predicted_category = self._predict_mask_category(image, segmentation, area_percentage)
        
        # Determine walkability relevance
        category_info = self.walkability_categories.get(predicted_category, {'impact': 'neutral', 'weight': 0.0})
        walkability_relevance = category_info['impact']
        
        return MaskAnalysis(
            mask_id=mask_id,
            area_pixels=area_pixels,
            area_percentage=area_percentage,
            predicted_category=predicted_category,
            confidence=0.7,  # Placeholder - would come from trained classifier
            walkability_relevance=walkability_relevance
        )
    
    def _predict_mask_category(self, image: np.ndarray, mask: np.ndarray, area_percentage: float) -> str:
        """
        Predict what category a mask represents using simple heuristics
        
        In a production system, this would be replaced with:
        1. A trained CNN classifier
        2. Integration with existing object detection models (YOLO, etc.)
        3. Semantic segmentation models trained on street scene data
        """
        # Extract masked region
        masked_region = image[mask]
        
        if len(masked_region) == 0:
            return 'unknown'
        
        # Simple color-based heuristics (very basic - needs improvement)
        avg_color = np.mean(masked_region, axis=0)
        
        # Heuristic rules based on color, position, and size
        height, width = image.shape[:2]
        mask_coords = np.where(mask)
        
        if len(mask_coords[0]) == 0:
            return 'unknown'
            
        avg_y = np.mean(mask_coords[0])
        avg_x = np.mean(mask_coords[1])
        
        # Bottom third of image + gray colors = likely sidewalk
        if avg_y > height * 0.66 and self._is_gray_ish(avg_color) and area_percentage > 5:
            return 'sidewalk'
        
        # Top third + blue/white = likely sky
        if avg_y < height * 0.33 and (avg_color[2] > 150 or np.mean(avg_color) > 200):
            return 'sky'
        
        # Green colors = vegetation
        if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2] and avg_color[1] > 100:
            return 'tree'
        
        # Small objects in middle/lower area = street furniture
        if 0.1 < area_percentage < 3 and avg_y > height * 0.4:
            return 'street_furniture'
        
        # Large dark areas in bottom = road
        if avg_y > height * 0.5 and np.mean(avg_color) < 80 and area_percentage > 10:
            return 'road'
        
        # Default to building for large areas
        if area_percentage > 15:
            return 'building'
        
        return 'unknown'
    
    def _is_gray_ish(self, color: np.ndarray) -> bool:
        """Check if color is grayish (potential concrete/sidewalk)"""
        r, g, b = color
        # Gray if colors are similar and not too bright/dark
        color_diff = max(r, g, b) - min(r, g, b)
        avg_brightness = np.mean(color)
        return color_diff < 30 and 50 < avg_brightness < 180
    
    def _extract_walkability_features(self, mask_analyses: List[MaskAnalysis], image_shape: Tuple) -> WalkabilityFeatures:
        """Extract high-level walkability features from mask analyses"""
        
        # Calculate coverage percentages
        sidewalk_coverage = sum(
            analysis.area_percentage for analysis in mask_analyses 
            if analysis.predicted_category == 'sidewalk'
        )
        
        vegetation_coverage = sum(
            analysis.area_percentage for analysis in mask_analyses 
            if analysis.predicted_category in ['tree', 'vegetation']
        )
        
        # Count specific features
        crosswalk_detected = any(
            analysis.predicted_category == 'crosswalk' for analysis in mask_analyses
        )
        
        street_furniture_count = sum(
            1 for analysis in mask_analyses 
            if analysis.predicted_category in ['bench', 'bus_stop', 'bike_rack', 'street_furniture']
        )
        
        lighting_elements = sum(
            1 for analysis in mask_analyses 
            if analysis.predicted_category in ['street_light', 'sign']
        )
        
        obstacles_detected = sum(
            1 for analysis in mask_analyses 
            if analysis.predicted_category in ['construction', 'barrier', 'debris']
        )
        
        # Calculate overall visual walkability score
        overall_score = self._calculate_visual_walkability_score(mask_analyses)
        
        return WalkabilityFeatures(
            sidewalk_coverage=sidewalk_coverage,
            crosswalk_detected=crosswalk_detected,
            street_furniture_count=street_furniture_count,
            vegetation_coverage=vegetation_coverage,
            lighting_elements=lighting_elements,
            obstacles_detected=obstacles_detected,
            overall_walkability_score=overall_score
        )
    
    def _calculate_visual_walkability_score(self, mask_analyses: List[MaskAnalysis]) -> float:
        """Calculate overall visual walkability score (0-100)"""
        score = 50.0  # Base score
        
        for analysis in mask_analyses:
            category_info = self.walkability_categories.get(
                analysis.predicted_category, 
                {'impact': 'neutral', 'weight': 0.0}
            )
            
            # Weight by area and category importance
            area_weight = min(analysis.area_percentage / 10.0, 1.0)  # Cap at 10% area
            impact = category_info['weight'] * area_weight * 100
            score += impact
        
        # Clamp to 0-100 range
        return max(0.0, min(100.0, score))

    def create_annotated_overlay(self, image: np.ndarray, masks: List[Dict], 
                                mask_analyses: List[MaskAnalysis]) -> np.ndarray:
        """Create overlay with walkability-focused annotations"""
        overlay = image.copy()
        
        # Color code by walkability impact
        colors = {
            'positive': [0, 255, 0],    # Green for positive features
            'negative': [255, 0, 0],    # Red for negative features  
            'neutral': [128, 128, 128], # Gray for neutral features
        }
        
        for mask, analysis in zip(masks, mask_analyses):
            seg = mask.get("segmentation")
            if seg is None:
                continue
                
            color = colors.get(analysis.walkability_relevance, [128, 128, 128])
            
            # Create colored mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[seg] = color
            
            # Apply with transparency
            mask_area = seg.astype(np.uint8) * 255
            mask_area = np.stack([mask_area] * 3, axis=-1)
            
            overlay = np.where(mask_area > 0, 
                              cv2.addWeighted(colored_mask, 0.3, overlay, 0.7, 0),
                              overlay)
        
        return overlay