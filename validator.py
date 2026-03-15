"""
Validator for NBA Court Optimization system.

This module validates model predictions against actual game simulations,
calculating error metrics and confidence intervals.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats

from src.game_simulator import GameSimulator


class Validator:
    """Validates model predictions against simulations"""
    
    def __init__(self, game_simulator: GameSimulator):
        """
        Initialize validator with game simulator
        
        Args:
            game_simulator: GameSimulator instance for running validation simulations
        """
        self.game_simulator = game_simulator
    
    def validate_prediction(self, 
                          three_pt_radius: float, 
                          baseline_width: float,
                          predicted_score: float, 
                          n_simulations: int = 100) -> Dict:
        """
        Validate prediction by running actual simulations
        
        Args:
            three_pt_radius: 3-point radius to validate (23.75-26.0 feet)
            baseline_width: Baseline width to validate (50.0-55.0 feet)
            predicted_score: Model's predicted combined score
            n_simulations: Number of simulations to run (default: 100)
            
        Returns:
            Dictionary with:
            - mean_simulated_score: Average combined score from simulations
            - std_simulated_score: Standard deviation of simulated scores
            - confidence_interval_95: (lower, upper) bounds for 95% CI
            - mae: Mean absolute error between prediction and simulation mean
            
        Raises:
            ValueError: If n_simulations < 10 (insufficient for meaningful CI)
        """
        # Validate input
        if n_simulations < 10:
            raise ValueError(
                f"Insufficient simulations: {n_simulations}. "
                "At least 10 simulations required for meaningful confidence intervals."
            )
        
        # Run simulations
        simulation_results = self.game_simulator.simulate_game(
            three_pt_radius=three_pt_radius,
            baseline_width=baseline_width,
            n_simulations=n_simulations
        )
        
        # Extract combined scores from simulation results
        # Each result is a tuple: (warriors_score, cavaliers_score)
        combined_scores = [warriors + cavaliers for warriors, cavaliers in simulation_results]
        
        # Calculate statistics
        mean_score = np.mean(combined_scores)
        std_score = np.std(combined_scores, ddof=1)  # Sample standard deviation
        
        # Calculate 95% confidence interval
        ci_lower, ci_upper = self.calculate_confidence_interval(
            combined_scores, 
            confidence=0.95
        )
        
        # Calculate mean absolute error
        mae = abs(predicted_score - mean_score)
        
        return {
            'mean_simulated_score': mean_score,
            'std_simulated_score': std_score,
            'confidence_interval_95': (ci_lower, ci_upper),
            'mae': mae
        }
    
    def calculate_confidence_interval(self, 
                                     scores: list, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for simulated scores
        
        Uses t-distribution for small sample sizes and normal distribution
        for large samples (n >= 30).
        
        Args:
            scores: List of simulated combined scores
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
            
        Raises:
            ValueError: If scores list is empty or confidence not in (0, 1)
        """
        # Validate input
        if not scores:
            raise ValueError("Cannot calculate confidence interval for empty scores list")
        
        if not (0 < confidence < 1):
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        # Convert to numpy array for calculations
        scores_array = np.array(scores)
        n = len(scores_array)
        
        # Calculate mean and standard error
        mean = np.mean(scores_array)
        std_error = stats.sem(scores_array)  # Standard error of the mean
        
        # Use t-distribution for confidence interval
        # (appropriate for any sample size, especially small samples)
        degrees_of_freedom = n - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
        
        # Calculate margin of error
        margin_of_error = t_critical * std_error
        
        # Calculate confidence interval bounds
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        return (lower_bound, upper_bound)
