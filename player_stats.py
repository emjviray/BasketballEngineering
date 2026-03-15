"""
PlayerStats module for NBA Court Optimization system.

This module manages player-specific statistics including EPPG values
(calculated dynamically), shooting percentages, and usage rates.

REFACTORED: This class now delegates to ShotDistribution for all player-specific
data, eliminating code duplication and ensuring consistency.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class PlayerStats:
    """
    Manages player-specific statistics for game simulation.
    
    This class delegates to ShotDistribution for player usage rates, minutes,
    shot distributions, and heatmaps. It provides a simplified interface for
    accessing player data and calculating EPPG values.
    
    REFACTORED: Now uses ShotDistribution as the single source of truth for
    all player-specific data.
    """
    
    def __init__(self, shot_distribution, eppg_calculator=None):
        """
        Initialize PlayerStats with ShotDistribution and EPPGCalculator.
        
        Args:
            shot_distribution: ShotDistribution instance containing player heatmaps,
                             usage rates, and minutes data
            eppg_calculator: EPPGCalculator instance for dynamic EPPG calculation
                           (optional, can be None if not yet implemented)
        """
        if shot_distribution is None:
            raise ValueError("ShotDistribution instance cannot be None")
        
        self.shot_distribution = shot_distribution
        self.eppg_calculator = eppg_calculator

    def get_player_eppg(self, player_name: str, three_pt_radius: float, 
                       baseline_width: float) -> float:
        """
        Calculate EPPG dynamically for any court configuration.

        EPPG is calculated using ShotDistribution's player heatmaps, which
        reclassify historical shots as 2-pointers or 3-pointers based on
        the court dimensions. This accounts for minutes played.

        Args:
            player_name: Name of the player
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)

        Returns:
            Expected Points Per Game for the player at this court configuration

        Raises:
            ValueError: If player not found in shot data
        """
        # If EPPGCalculator is available, use it for accurate calculation
        if self.eppg_calculator is not None:
            eppg_data = self.eppg_calculator.calculate_eppg_for_configuration(
                three_pt_radius, baseline_width
            )
            player_eppg = eppg_data[eppg_data['PLAYER_NAME'] == player_name]
            if not player_eppg.empty:
                return float(player_eppg['eppg'].iloc[0])

        # Fallback: Use ShotDistribution's heatmap-based calculation
        # Get expected points per shot from ShotDistribution
        expected_points_per_shot = self.shot_distribution.get_player_expected_points(
            player_name, three_pt_radius, baseline_width
        )

        # Get minutes per game from ShotDistribution
        minutes = self.shot_distribution.get_player_minutes(player_name)

        # Get shots per minute (estimated from historical data)
        shots_per_minute = self.get_player_shots_per_minute(player_name)

        # EPPG = shots per minute * minutes per game * expected points per shot
        return shots_per_minute * minutes * expected_points_per_shot
    
    def get_player_minutes(self, player_name: str) -> float:
        """
        Get player's minutes per game.

        Delegates to ShotDistribution for accurate minutes data.

        Args:
            player_name: Name of the player

        Returns:
            Minutes per game as a float

        Raises:
            ValueError: If player not found in shot data
        """
        return self.shot_distribution.get_player_minutes(player_name)
    
    def get_player_shots_per_minute(self, player_name: str) -> float:
        """
        Calculate shots per minute for a player.

        Uses ShotDistribution's heatmap data to calculate shot frequency.

        Args:
            player_name: Name of the player

        Returns:
            Shots per minute as a float

        Raises:
            ValueError: If player not found in shot data
        """
        # Get player heatmap from ShotDistribution
        if player_name not in self.shot_distribution.player_heatmaps:
            raise ValueError(f"Player '{player_name}' not found in shot data")

        heatmap = self.shot_distribution.player_heatmaps[player_name]

        # Calculate total shots from heatmap
        total_shots = sum(cell['count'] for cell in heatmap.values())

        # Get minutes per game
        minutes = self.get_player_minutes(player_name)

        if minutes == 0:
            return 0.0

        # Get games played from shot data
        shot_data = self.shot_distribution.shot_data
        player_shots = shot_data[shot_data['PLAYER_NAME'] == player_name]
        games_played = player_shots['GAME_ID'].nunique() if 'GAME_ID' in player_shots.columns else 1

        # Shots per minute = total shots / (games * minutes per game)
        return total_shots / (games_played * minutes)
    
    def get_player_points_per_minute(self, player_name: str, three_pt_radius: float, 
                                     baseline_width: float) -> float:
        """
        Calculate points per minute for a player at a given court configuration.
        
        Args:
            player_name: Name of the player
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            
        Returns:
            Points per minute as a float
            
        Raises:
            ValueError: If player not found in shot data
        """
        eppg = self.get_player_eppg(player_name, three_pt_radius, baseline_width)
        minutes = self.get_player_minutes(player_name)
        
        if minutes == 0:
            return 0.0
        
        return eppg / minutes
    
    def get_player_usage_rate(self, player_name: str) -> float:
        """
        Get player usage rate from historical shot attempts.

        Delegates to ShotDistribution for accurate usage rate data.
        Usage rate represents the proportion of team shot attempts taken by
        this player.

        Args:
            player_name: Name of the player

        Returns:
            Usage rate as a float between 0.0 and 1.0

        Raises:
            ValueError: If player not found in shot data
        """
        return self.shot_distribution.get_player_usage_rate(player_name)
    
    def get_player_shot_distribution(self, player_name: str, 
                                     three_pt_radius: float = 23.75,
                                     baseline_width: float = 50.0) -> Dict[str, float]:
        """
        Get player shot distribution based on court dimensions.

        Delegates to ShotDistribution which uses player heatmaps to reclassify
        shots based on the court configuration.

        Returns the frequency of each shot type for the player:
        - rim: Proportion of shots at the rim
        - midrange: Proportion of mid-range shots
        - corner3: Proportion of corner 3-pointers
        - above_break3: Proportion of above-break 3-pointers

        Args:
            player_name: Name of the player
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)

        Returns:
            Dictionary with shot type frequencies (sum to 1.0)

        Raises:
            ValueError: If player not found in shot data
        """
        return self.shot_distribution.get_player_shot_distribution(
            player_name, three_pt_radius, baseline_width
        )
    
    def get_team_players(self, team_name: str) -> List[str]:
        """
        List all players for a specific team.

        Delegates to ShotDistribution for team roster data.

        Args:
            team_name: Name of the team ('Golden State Warriors' or 'Cleveland Cavaliers')

        Returns:
            List of player names on the team

        Raises:
            ValueError: If team not found in shot data
        """
        return self.shot_distribution.get_team_players(team_name)
    
    def get_all_players(self) -> List[str]:
        """
        Get list of all players in the dataset.

        Returns:
            List of all unique player names
        """
        return self.shot_distribution.shot_data['PLAYER_NAME'].unique().tolist()
    
    def get_player_team(self, player_name: str) -> str:
        """
        Get the team for a specific player.

        Args:
            player_name: Name of the player

        Returns:
            Team name

        Raises:
            ValueError: If player not found in shot data
        """
        shot_data = self.shot_distribution.shot_data
        player_data = shot_data[shot_data['PLAYER_NAME'] == player_name]

        if player_data.empty:
            raise ValueError(f"Player '{player_name}' not found in shot data")

        return player_data['TEAM_NAME'].iloc[0]
