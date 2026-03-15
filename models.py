"""
Data models for NBA Court Optimization system.

This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass
class CourtConfiguration:
    """Represents a basketball court configuration"""
    three_pt_radius: float  # 23.75 - 26.0 feet
    baseline_width: float   # 50.0 - 55.0 feet
    
    def __post_init__(self):
        """Validate dimensions are within acceptable ranges"""
        if not (23.75 <= self.three_pt_radius <= 26.0):
            raise ValueError("3-point radius must be between 23.75 and 26.0 feet")
        if not (50.0 <= self.baseline_width <= 55.0):
            raise ValueError("Baseline width must be between 50.0 and 55.0 feet")
    
    def calculate_corner3_distance(self) -> float:
        """Calculate corner 3-point distance based on geometry"""
        # Distance from basket to corner 3 point line
        hoop_to_baseline = 4.0
        return min(self.three_pt_radius, 
                  math.sqrt(self.three_pt_radius**2 - hoop_to_baseline**2))


@dataclass
class TeamStatistics:
    """Team-level statistics for game simulation"""
    team_name: str
    pace: float                    # Possessions per game
    off_reb_rate: float           # Offensive rebound rate
    def_reb_rate: float           # Defensive rebound rate
    turnover_rate: float          # Turnover rate
    free_throw_rate: float        # Free throw attempts per FGA
    
    # Shot distribution
    rim_attempt_share: float      # Proportion of shots at rim
    midrange_share: float         # Proportion of midrange shots
    corner3_pa: float            # Corner 3 attempts per game
    above_break3_pa: float       # Above-break 3 attempts per game
    
    # Shooting efficiency
    threepar: float              # 3-point attempt rate
    team_rim_rate: float         # Rim FG%
    team_corner3_rate: float     # Corner 3 FG%
    
    # Defensive stats
    opp_3par_allowed: float      # Opponent 3PA rate allowed
    opp_rim_fg_allowed: float    # Opponent rim FG% allowed
    switch_rate: float           # Defensive switch rate


@dataclass
class PlayerStatistics:
    """Player-specific statistics"""
    player_name: str
    team: str                    # 'Warriors' or 'Cavaliers'
    eppg: float                  # Expected points per game
    usage_rate: float            # Proportion of team possessions used
    
    # Shot type preferences (sum to 1.0)
    rim_frequency: float
    midrange_frequency: float
    corner3_frequency: float
    above_break3_frequency: float
    
    # Shooting percentages by zone
    rim_fg_pct: float
    midrange_fg_pct: float
    corner3_fg_pct: float
    above_break3_fg_pct: float
    free_throw_pct: float


@dataclass
class GameResult:
    """Result of a simulated game"""
    warriors_score: int
    cavaliers_score: int
    combined_score: int
    court_config: CourtConfiguration
    
    # Detailed statistics
    warriors_possessions: int
    cavaliers_possessions: int
    warriors_shot_breakdown: Dict[str, int]  # Shots by type
    cavaliers_shot_breakdown: Dict[str, int]


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    optimal_config: CourtConfiguration
    predicted_score: float
    score_difference: float  # Difference from target (200)
    
    # Individual team predictions
    warriors_predicted: float
    cavaliers_predicted: float
    
    # Validation metrics
    validation_mean: float
    validation_std: float
    confidence_interval: Tuple[float, float]
    
    # Alternative configurations
    top_5_configs: List[Tuple[CourtConfiguration, float]]
