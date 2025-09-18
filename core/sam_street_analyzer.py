"""
SAM-based street view analysis for walkability features
Uses position and size heuristics to categorize SAM segments
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from segment_anything import sam_model_registry, SamPredictor

@dataclass
class StreetSegmentAnalysis:
    """Analysis results from SAM street view segmentation"""
    road_coverage: float  # Percentage of image that's road
    footpath_coverage: float  # Percentage that's sidewalk/footpath
    building_coverage: float  # Percentage that's buildings
    vegetation_coverage: float  # Percentage that's trees/vegetation
    sky_coverage: float  # Percentage that's sky
    vehicle_count: int  # Number of detected vehicles
    sign_count: int  # Number of detected signs
    total_segments: int  # Total number of segments found
    
    def get_walkability_indicators(self) -> Dict[str, float]:
        """Get walkability-relevant indicators from visual analysis"""
        return {
            'footpath_availability': self.footpath_coverage,  # Higher = better
            'road_dominance': self.road_coverage,  # Lower might be better for pedestrians
            'urban_density': self.building_coverage,  # Moderate levels good
            'green_space': self.vegetation_coverage,  # Higher = better
            'traffic_density': self.vehicle_count,  # Lower = better for walking
            'wayfinding_support': self.sign_count  # Moderate levels good
        }

class SAMStreetAnalyzer:
    """Analyzes street view images using SAM with position-based heuristics"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "vit_b"):
        """
        Initialize SAM model for street analysis
        
        Args:
            model_path: Path to SAM checkpoint (if None, will need to be provided later)
            model_type: SAM model type (vit_b, vit_l, vit_h)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.predictor = None
        
        if model_path:
            self.load_model(model_path, model_type)
    
    def load_model(self, model_path: str, model_type: str = None):
        """Load SAM model"""
        if model_type:
            self.model_type = model_type
            
        sam = sam_model_registry[self.model_type](checkpoint=model_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
    
    def generate_grid_points(self, image_shape: Tuple[int, int], grid_size: int = 64) -> np.ndarray:
        """Generate grid points for systematic segmentation"""
        height, width = image_shape[:2]
        points = []
        
        # Generate grid points, avoiding edges
        for y in range(grid_size, height - grid_size, grid_size):
            for x in range(grid_size, width - grid_size, grid_size):
                points.append([x, y])
        
        return np.array(points)
    
    def analyze_street_view(self, image: np.ndarray, confidence_threshold: float = 0.8) -> Tuple[StreetSegmentAnalysis, Dict[str, np.ndarray]]:
        """
        Analyze street view image for walkability-relevant features
        
        Args:
            image: RGB image array
            confidence_threshold: Minimum confidence for keeping segments
            
        Returns:
            Tuple of (StreetSegmentAnalysis, coverage_masks dict)
        """
        if self.predictor is None:
            raise ValueError("SAM model not loaded. Call load_model() first.")
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Generate grid points
        points = self.generate_grid_points(image.shape, grid_size=64)
        
        # Initialize coverage masks
        coverage_masks = {
            'road': np.zeros((height, width), dtype=bool),
            'footpath': np.zeros((height, width), dtype=bool),
            'building': np.zeros((height, width), dtype=bool),
            'tree': np.zeros((height, width), dtype=bool),
            'sky': np.zeros((height, width), dtype=bool),
            'vehicle': np.zeros((height, width), dtype=bool),
            'sign': np.zeros((height, width), dtype=bool)
        }
        
        vehicle_count = 0
        sign_count = 0
        total_segments = 0
        
        # Process each grid point
        for point in points:
            try:
                # Predict mask for this point
                masks_pred, scores_pred, _ = self.predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([1]),  # Foreground point
                    multimask_output=True
                )
                
                # Get the best mask
                best_idx = np.argmax(scores_pred)
                mask = masks_pred[best_idx]
                score = scores_pred[best_idx]
                
                # Only keep high-confidence masks
                if score > confidence_threshold:
                    category = self._categorize_segment(mask, point, image)
                    
                    # Update coverage masks
                    if category in coverage_masks:
                        coverage_masks[category] = np.logical_or(coverage_masks[category], mask)
                    
                    # Count discrete objects
                    if category == 'vehicle':
                        vehicle_count += 1
                    elif category == 'sign':
                        sign_count += 1
                    
                    total_segments += 1
                    
            except Exception:
                # Skip problematic points
                continue
        
        # Calculate coverage percentages
        road_coverage = (np.sum(coverage_masks['road']) / total_pixels) * 100
        footpath_coverage = (np.sum(coverage_masks['footpath']) / total_pixels) * 100
        building_coverage = (np.sum(coverage_masks['building']) / total_pixels) * 100
        vegetation_coverage = (np.sum(coverage_masks['tree']) / total_pixels) * 100
        sky_coverage = (np.sum(coverage_masks['sky']) / total_pixels) * 100
        
        analysis = StreetSegmentAnalysis(
            road_coverage=road_coverage,
            footpath_coverage=footpath_coverage,
            building_coverage=building_coverage,
            vegetation_coverage=vegetation_coverage,
            sky_coverage=sky_coverage,
            vehicle_count=vehicle_count,
            sign_count=sign_count,
            total_segments=total_segments
        )
        
        return analysis, coverage_masks
    
    def _categorize_segment(self, mask: np.ndarray, point: np.ndarray, image: np.ndarray) -> str:
        """
        Categorize a segment using position and size heuristics
        Based on the street view segmentation code you found
        """
        height, width = mask.shape
        
        # Calculate mask properties
        mask_area = np.sum(mask)
        if mask_area == 0:
            return 'unknown'
        
        # Find mask center
        mask_coords = np.where(mask)
        if len(mask_coords[0]) == 0:
            return 'unknown'
            
        mask_center_y = np.mean(mask_coords[0])
        mask_center_x = np.mean(mask_coords[1])
        
        # Position-based categorization (from your found code)
        if mask_center_y > height * 0.7:  # Bottom third
            if mask_center_x < width * 0.3 or mask_center_x > width * 0.7:
                return 'footpath'  # Sides of bottom = sidewalks
            else:
                return 'road'  # Center of bottom = road
        
        elif mask_center_y < height * 0.3:  # Top third
            return 'sky'
        
        elif mask_center_y < height * 0.6:  # Middle area
            if mask_area > 5000:  # Large segments
                return 'building'
            else:
                return 'sign'
        
        else:
            # Size-based categorization for remaining areas
            if mask_area > 3000:
                return 'vehicle'
            elif mask_area > 1000:
                return 'tree'
            else:
                return 'sign'
    
    def create_segmentation_overlay(self, image: np.ndarray, coverage_masks: Dict[str, np.ndarray], alpha: float = 0.6) -> np.ndarray:
        """
        Create colored segmentation overlay showing different categories
        
        Args:
            image: Original RGB image
            coverage_masks: Dictionary of category masks
            alpha: Transparency for overlay
            
        Returns:
            Image with colored segmentation overlay
        """
        # Define colors for each category (similar to original street view segmenter)
        category_colors = {
            'road': (128, 128, 128),        # Gray
            'footpath': (0, 255, 0),        # Green  
            'building': (255, 0, 0),        # Red
            'tree': (0, 128, 0),            # Dark Green
            'sky': (135, 206, 235),         # Sky Blue
            'vehicle': (0, 0, 255),         # Blue
            'sign': (255, 255, 0)           # Yellow
        }
        
        # Create combined colored overlay
        overlay = np.zeros_like(image)
        
        for category, mask in coverage_masks.items():
            if np.any(mask) and category in category_colors:
                color = category_colors[category]
                overlay[mask] = color
        
        # Blend with original image
        result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        # Add legend
        result_with_legend = self._add_legend(result, category_colors, coverage_masks)
        
        return result_with_legend
    
    def _add_legend(self, image: np.ndarray, category_colors: Dict[str, Tuple], coverage_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Add legend showing categories and their colors"""
        import cv2
        
        # Create legend on the right side
        legend_width = 200
        legend_height = len(category_colors) * 25 + 20
        
        # Extend image to add legend space
        height, width = image.shape[:2]
        extended_image = np.zeros((height, width + legend_width, 3), dtype=np.uint8)
        extended_image[:, :width] = image
        extended_image[:, width:] = (50, 50, 50)  # Dark background for legend
        
        # Add legend items
        y_offset = 20
        for category, color in category_colors.items():
            if np.any(coverage_masks.get(category, False)):  # Only show categories that were found
                # Draw color box
                cv2.rectangle(extended_image, 
                            (width + 10, y_offset), 
                            (width + 30, y_offset + 15), 
                            color, -1)
                
                # Add text
                cv2.putText(extended_image, 
                          category.title(), 
                          (width + 35, y_offset + 12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.4, (255, 255, 255), 1)
                
                y_offset += 25
        
        return extended_image