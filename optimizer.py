"""
Optimizer module for NBA Court Optimization system.

This module implements the optimization layer that searches for optimal court
dimensions using a trained neural network model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models import CourtConfiguration, OptimizationResult
from src.neural_network_model import NeuralNetworkModel
from src.feature_engineering import FeatureEngineering


class Optimizer:
    """Finds optimal court dimensions using trained neural network"""
    
    def __init__(self, model: NeuralNetworkModel, feature_engineering: FeatureEngineering, 
                 warriors_stats: Dict = None, cavaliers_stats: Dict = None):
        """
        Initialize optimizer with trained model and feature engineering.
        
        Args:
            model: Trained NeuralNetworkModel instance
            feature_engineering: FeatureEngineering instance (must be fitted)
            warriors_stats: Warriors team statistics (optional, for accurate feature construction)
            cavaliers_stats: Cavaliers team statistics (optional, for accurate feature construction)
        
        Raises:
            RuntimeError: If model is not trained (has no history)
        """
        if not hasattr(model, 'history') or not model.history.get('train_loss'):
            raise RuntimeError("Model must be trained before optimization")
        
        self.model = model
        self.feature_engineering = feature_engineering
        self.warriors_stats = warriors_stats
        self.cavaliers_stats = cavaliers_stats
    
    def optimize(self, target_score: float = 200.0,
                three_pt_range: Tuple[float, float] = (23.75, 26.0),
                baseline_range: Tuple[float, float] = (50.0, 55.0),
                grid_resolution: float = 0.25) -> Dict:
        """
        Search for optimal court dimensions using grid search.
        
        Args:
            target_score: Target combined score (default: 200.0)
            three_pt_range: (min, max) for 3-point radius (default: 23.75-26.0)
            baseline_range: (min, max) for baseline width (default: 50.0-55.0)
            grid_resolution: Step size for grid search (default: 0.25)
            
        Returns:
            Dictionary with:
            - optimal_3pt_radius: Best 3-point radius
            - optimal_baseline_width: Best baseline width
            - predicted_score: Predicted combined score
            - score_difference: Difference from target
            - top_5_configs: Top 5 configurations as list of tuples
        
        Raises:
            ValueError: If target_score is not positive
            ValueError: If ranges are invalid
        """
        # Validate inputs
        if target_score <= 0:
            raise ValueError("Target score must be positive")
        
        if three_pt_range[0] >= three_pt_range[1]:
            raise ValueError("Invalid 3-point radius range")
        
        if baseline_range[0] >= baseline_range[1]:
            raise ValueError("Invalid baseline width range")
        
        # Warn if target score is unrealistic
        if target_score < 100 or target_score > 300:
            print(f"Warning: Target score {target_score} may be unrealistic (typical range: 100-300)")
        
        # Generate grid of configurations
        three_pt_values = np.arange(three_pt_range[0], three_pt_range[1] + grid_resolution/2, grid_resolution)
        baseline_values = np.arange(baseline_range[0], baseline_range[1] + grid_resolution/2, grid_resolution)
        
        # Clip values to ensure they stay within bounds (handle floating point precision issues)
        three_pt_values = np.clip(three_pt_values, three_pt_range[0], three_pt_range[1])
        baseline_values = np.clip(baseline_values, baseline_range[0], baseline_range[1])
        
        # Store all configurations and their predicted scores
        all_configs = []
        
        for three_pt_radius in three_pt_values:
            for baseline_width in baseline_values:
                # Evaluate this configuration
                predicted_score = self.evaluate_configuration(three_pt_radius, baseline_width)
                
                # Calculate difference from target
                score_diff = abs(predicted_score - target_score)
                
                # Store configuration
                config = CourtConfiguration(
                    three_pt_radius=float(three_pt_radius),
                    baseline_width=float(baseline_width)
                )
                all_configs.append((config, predicted_score, score_diff))
        
        # Check if we have any valid configurations
        if not all_configs:
            raise RuntimeError("No valid configurations found in search space")
        
        # Sort by score difference (ascending)
        all_configs.sort(key=lambda x: x[2])
        
        # Get top 5 configurations
        top_5 = [(config, score) for config, score, _ in all_configs[:5]]
        
        # Get optimal configuration (first in sorted list)
        optimal_config, optimal_score, optimal_diff = all_configs[0]
        
        return {
            'optimal_3pt_radius': optimal_config.three_pt_radius,
            'optimal_baseline_width': optimal_config.baseline_width,
            'predicted_score': optimal_score,
            'score_difference': optimal_diff,
            'top_5_configs': top_5
        }
    
    def evaluate_configuration(self, three_pt_radius: float, 
                              baseline_width: float) -> float:
        """
        Predict combined score for a specific court configuration.
        
        FIX: Use the same feature construction pipeline as training.
        This ensures prediction inputs match training inputs.
        
        Args:
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            
        Returns:
            Predicted combined score
            
        Raises:
            ValueError: If court dimensions are out of valid range
        """
        # Validate dimensions
        if not (23.75 <= three_pt_radius <= 26.0):
            raise ValueError(f"3-point radius {three_pt_radius} out of range [23.75, 26.0]")
        
        if not (50.0 <= baseline_width <= 55.0):
            raise ValueError(f"Baseline width {baseline_width} out of range [50.0, 55.0]")
        
        # FIX: Use actual team statistics if provided, otherwise use defaults
        if self.warriors_stats is not None and self.cavaliers_stats is not None:
            # Use actual team statistics from training
            config_data = pd.DataFrame([{
                'r_3pt_radius': three_pt_radius,
                'baseline_width': baseline_width,
                'team': 'Warriors2016',
                'pace': self.warriors_stats['pace'],
                'off_reb_rate': self.warriors_stats['off_reb_rate'],
                'def_reb_rate': self.warriors_stats['def_reb_rate'],
                'turnover_rate': self.warriors_stats['turnover_rate'],
                'free_throw_rate': self.warriors_stats['free_throw_rate'],
                'rim_attempt_share': self.warriors_stats['rim_attempt_share'],
                'midrange_share': self.warriors_stats['midrange_share'],
                'corner3_pa': self.warriors_stats['corner3_pa'],
                'above_break3_pa': self.warriors_stats['above_break3_pa'],
                'threepar': self.warriors_stats['threepar'],
                'team_rim_rate': self.warriors_stats['team_rim_rate'],
                'team_corner3_rate': self.warriors_stats['team_corner3_rate'],
                'opp_3par_allowed': self.warriors_stats['opp_3par_allowed'],
                'opp_rim_fg_allowed': self.warriors_stats['opp_rim_fg_allowed']
            }])
        else:
            # Fallback to placeholder values (not recommended)
            config_data = pd.DataFrame([{
                'r_3pt_radius': three_pt_radius,
                'baseline_width': baseline_width,
                'team': 'Warriors2016',
                'corner3_pa': 5.0,
                'above_break3_pa': 15.0,
                'rim_attempt_share': 0.35,
                'midrange_share': 0.15,
                'pace': 95.0,
                'off_reb_rate': 0.25,
                'turnover_rate': 0.14,
                'def_reb_rate': 0.75,
                'opp_3par_allowed': 0.35,
                'opp_rim_fg_allowed': 0.60,
                'threepar': 0.35,
                'team_rim_rate': 0.65,
                'team_corner3_rate': 0.38,
                'free_throw_rate': 0.25
            }])
        
        # Extract features using the SAME pipeline as training
        # This will calculate geometric features automatically
        features = self.feature_engineering.extract_features(config_data, eppg_data=None)
        
        # Normalize features using stored parameters from training
        normalized_features = self.feature_engineering.normalize_features(features, fit=False)
        
        # Predict score
        predicted_score = self.model.predict(normalized_features)[0]
        
        return float(predicted_score)