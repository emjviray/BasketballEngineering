"""
FeatureEngineering module for NBA Court Optimization system.

This module transforms raw data into model features, including normalization
and geometric feature calculations.
"""

import numpy as np
import pandas as pd
from typing import Optional
import math


class FeatureEngineering:
    """Transforms raw data into model features"""
    
    def __init__(self):
        """Initialize feature engineering with storage for normalization parameters"""
        self.feature_means_ = None
        self.feature_stds_ = None
        self.feature_names_ = None
    
    def extract_features(self, grid_data: pd.DataFrame, 
                        eppg_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Extract and normalize features for model training.
        
        Features include:
        - Court dimensions: r_3pt_radius, baseline_width
        - Geometric features: corner3_distance, arc_length_inbounds, area_inside_arc
        - Shot distributions: corner3_pa, above_break3_pa, rim_attempt_share, midrange_share
        - Team stats: pace, off_reb_rate, turnover_rate, def_reb_rate
        - Defensive stats: opp_3par_allowed, opp_rim_fg_allowed
        - Player EPPG aggregates: team_total_eppg, top_player_eppg (if eppg_data provided)
        
        Args:
            grid_data: DataFrame with team stats and court dimensions
            eppg_data: Optional DataFrame with player EPPG values
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Merge EPPG data if provided
        if eppg_data is not None:
            data = self.merge_eppg_with_grid(grid_data, eppg_data)
        else:
            data = grid_data.copy()
        
        # Define feature columns to extract
        feature_columns = [
            # Court dimensions
            'r_3pt_radius',
            'baseline_width',
            # Geometric features (calculate if not present)
            'corner3_distance',
            'arc_length_inbounds',
            'area_inside_arc',
            # Shot distributions
            'corner3_pa',
            'above_break3_pa',
            'rim_attempt_share',
            'midrange_share',
            # Team stats
            'pace',
            'off_reb_rate',
            'turnover_rate',
            'def_reb_rate',
            # Defensive stats
            'opp_3par_allowed',
            'opp_rim_fg_allowed'
        ]
        
        # Add EPPG aggregates if available
        if 'team_total_eppg' in data.columns:
            feature_columns.append('team_total_eppg')
        if 'top_player_eppg' in data.columns:
            feature_columns.append('top_player_eppg')
        
        # Calculate geometric features if not present
        if 'corner3_distance' not in data.columns:
            data['corner3_distance'] = data.apply(
                lambda row: self._calculate_corner3_distance(
                    row['r_3pt_radius'], 
                    row.get('baseline_width', 50.0)
                ), axis=1
            )
        
        if 'arc_length_inbounds' not in data.columns:
            data['arc_length_inbounds'] = data.apply(
                lambda row: self._calculate_arc_length(
                    row['r_3pt_radius'],
                    row.get('baseline_width', 50.0)
                ), axis=1
            )
        
        if 'area_inside_arc' not in data.columns:
            data['area_inside_arc'] = data.apply(
                lambda row: self._calculate_area_inside_arc(
                    row['r_3pt_radius'],
                    row.get('baseline_width', 50.0)
                ), axis=1
            )
        
        # Extract features
        features = data[feature_columns].values
        
        # Store feature names for reference
        self.feature_names_ = feature_columns
        
        return features
    
    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Apply standardization to features (zero mean, unit variance).
        
        Args:
            features: Feature matrix (n_samples, n_features)
            fit: If True, calculate and store mean/std. If False, use stored values.
        
        Returns:
            Normalized feature matrix
        """
        if fit:
            # Calculate mean and standard deviation
            self.feature_means_ = np.mean(features, axis=0)
            self.feature_stds_ = np.std(features, axis=0)
            
            # Handle zero standard deviation (constant features)
            self.feature_stds_[self.feature_stds_ == 0] = 1.0
        
        # Apply standardization
        normalized = (features - self.feature_means_) / self.feature_stds_
        
        return normalized
    
    def merge_eppg_with_grid(self, grid_data: pd.DataFrame, 
                            eppg_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine calculated EPPG with grid data.
        
        Aggregates player EPPG values by team and court configuration,
        then merges with grid data.
        
        Args:
            grid_data: DataFrame with team stats and court dimensions
            eppg_data: DataFrame with columns: player, team, r, w, EPPG
                      where r is 3pt radius and w is baseline width
        
        Returns:
            Merged DataFrame with EPPG aggregates added
        """
        # Create a copy to avoid modifying original
        merged = grid_data.copy()
        
        # Aggregate EPPG by team and court configuration
        # Calculate total team EPPG and top player EPPG
        eppg_agg = eppg_data.groupby(['team', 'r', 'w']).agg({
            'EPPG': ['sum', 'max']
        }).reset_index()
        
        # Flatten column names
        eppg_agg.columns = ['team', 'r', 'w', 'team_total_eppg', 'top_player_eppg']
        
        # Merge with grid data
        # Match on team and court dimensions
        merged = merged.merge(
            eppg_agg,
            left_on=['team', 'r_3pt_radius', 'baseline_width'],
            right_on=['team', 'r', 'w'],
            how='left'
        )
        
        # Drop duplicate columns from merge
        if 'r' in merged.columns:
            merged = merged.drop(columns=['r'])
        if 'w' in merged.columns:
            merged = merged.drop(columns=['w'])
        
        # Handle missing EPPG values (fill with 0 if no players found)
        merged['team_total_eppg'] = merged['team_total_eppg'].fillna(0.0)
        merged['top_player_eppg'] = merged['top_player_eppg'].fillna(0.0)
        
        return merged
    
    def _calculate_corner3_distance(self, three_pt_radius: float, 
                                   baseline_width: float) -> float:
        """
        Calculate corner 3-point distance based on geometry.
        
        Args:
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
        
        Returns:
            Corner 3-point distance in feet
        """
        hoop_to_baseline = 4.0
        # Corner 3 is the minimum of the radius and the distance from basket
        # to the corner along the baseline
        return min(three_pt_radius, 
                  math.sqrt(three_pt_radius**2 - hoop_to_baseline**2))
    
    def _calculate_arc_length(self, three_pt_radius: float, 
                             baseline_width: float) -> float:
        """
        Calculate the arc length of the 3-point line that is in bounds.
        
        Args:
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
        
        Returns:
            Arc length in feet
        """
        hoop_to_baseline = 4.0
        
        # Calculate the angle where the arc meets the baseline
        # Using geometry: the arc extends from one corner to the other
        corner_distance = self._calculate_corner3_distance(three_pt_radius, baseline_width)
        
        # Calculate angle from center to corner
        # If corner distance equals radius, angle is 90 degrees
        # Otherwise, use inverse sine
        if corner_distance >= three_pt_radius - 0.01:  # Nearly equal
            angle_rad = math.pi / 2
        else:
            # Angle from vertical to the point where arc meets baseline
            angle_rad = math.asin(hoop_to_baseline / three_pt_radius)
        
        # Arc length is radius * angle, and we have two symmetric sides
        # Total angle is pi minus 2 * angle_to_corner
        total_angle = math.pi - 2 * angle_rad
        arc_length = three_pt_radius * total_angle
        
        return arc_length
    
    def _calculate_area_inside_arc(self, three_pt_radius: float, 
                                  baseline_width: float) -> float:
        """
        Calculate the area inside the 3-point arc.
        
        Args:
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
        
        Returns:
            Area in square feet
        """
        hoop_to_baseline = 4.0
        
        # Area is approximately a circular sector minus the area behind the basket
        # Simplified calculation: semicircle area minus the area cut off by baseline
        
        # Calculate the angle where the arc meets the baseline
        corner_distance = self._calculate_corner3_distance(three_pt_radius, baseline_width)
        
        if corner_distance >= three_pt_radius - 0.01:
            angle_rad = math.pi / 2
        else:
            angle_rad = math.asin(hoop_to_baseline / three_pt_radius)
        
        # Total angle of the arc
        total_angle = math.pi - 2 * angle_rad
        
        # Area of circular sector
        sector_area = 0.5 * three_pt_radius**2 * total_angle
        
        # Add the rectangular area behind the arc to the baseline
        # This is the area between the arc and the baseline
        rect_width = 2 * math.sqrt(three_pt_radius**2 - hoop_to_baseline**2)
        rect_height = hoop_to_baseline
        rect_area = rect_width * rect_height
        
        total_area = sector_area + rect_area
        
        return total_area
