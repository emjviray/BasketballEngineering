"""
Shot distribution modeling for NBA Court Optimization system.

This module models how shot selection changes based on court dimensions using
player-specific spatial heatmaps. Each player has unique shot location preferences
captured in a grid-based heatmap system.
"""

from typing import Dict, Tuple, List
import math
import pandas as pd
import numpy as np


class ShotDistribution:
    """
    Models shot selection based on court dimensions using player-specific spatial heatmaps.
    
    This class creates grid-based heatmaps for each player showing WHERE they shoot from
    on the court. When court dimensions change, it reclassifies shots in each zone as
    2PT or 3PT and adjusts shot probabilities accordingly.
    """
    
    def __init__(self, shot_data: pd.DataFrame, grid_size: float = 2.0):
        """
        Initialize ShotDistribution with player shot data.
        
        Args:
            shot_data: DataFrame with columns: PLAYER_NAME, TEAM_NAME, LOC_X, LOC_Y,
                      SHOT_MADE_FLAG, GAME_ID (optional for minutes tracking).
                      LOC_X and LOC_Y are in tenths of feet with origin at the basket.
            grid_size: Size of each grid cell in feet (default 2.0 feet)
            
        Raises:
            ValueError: If shot_data is missing required columns
        """
        # Validate required columns
        required_columns = ['PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
        missing = [col for col in required_columns if col not in shot_data.columns]
        if missing:
            raise ValueError(f"Shot data missing required columns: {missing}")
        
        # Filter for Warriors and Cavaliers only
        self.shot_data = shot_data[
            shot_data['TEAM_NAME'].isin(['Golden State Warriors', 'Cleveland Cavaliers'])
        ].copy()
        
        if len(self.shot_data) == 0:
            raise ValueError("No shot data found for Warriors or Cavaliers")
        
        self.grid_size = grid_size
        
        # NBA standard court dimensions for reference
        self.standard_3pt_radius = 23.75  # feet
        self.standard_baseline_width = 50.0  # feet
        
        # Convert LOC_X and LOC_Y from tenths of feet to feet
        self.shot_data['LOC_X_FEET'] = self.shot_data['LOC_X'] / 10.0
        self.shot_data['LOC_Y_FEET'] = self.shot_data['LOC_Y'] / 10.0
        
        # Remove rows with missing location data
        self.shot_data = self.shot_data.dropna(subset=['LOC_X_FEET', 'LOC_Y_FEET', 'SHOT_MADE_FLAG'])
        
        if len(self.shot_data) == 0:
            raise ValueError("No valid shot data after removing missing values")
        
        # Build player heatmaps
        self.player_heatmaps = self._build_player_heatmaps()
        
        # Calculate and store usage rates and minutes per player
        self.player_usage_rates = self._calculate_usage_rates()
        self.player_minutes = self._calculate_minutes_per_game()
    
    def _build_player_heatmaps(self) -> Dict[str, Dict[Tuple[int, int], Dict]]:
        """
        Build spatial heatmaps for each player.
        
        Returns:
            Dictionary mapping player_name -> {(grid_x, grid_y): {
                'count': number of shots,
                'made': number of made shots,
                'fg_pct': field goal percentage,
                'avg_distance': average distance from basket
            }}
        """
        heatmaps = {}
        
        for player_name in self.shot_data['PLAYER_NAME'].unique():
            player_shots = self.shot_data[self.shot_data['PLAYER_NAME'] == player_name]
            heatmap = {}
            
            for _, shot in player_shots.iterrows():
                # Convert location to grid coordinates using floor for correct binning
                grid_x = math.floor(shot['LOC_X_FEET'] / self.grid_size)
                grid_y = math.floor(shot['LOC_Y_FEET'] / self.grid_size)
                grid_cell = (grid_x, grid_y)
                
                # Calculate distance from basket
                distance = math.sqrt(shot['LOC_X_FEET']**2 + shot['LOC_Y_FEET']**2)
                
                # Initialize or update grid cell
                if grid_cell not in heatmap:
                    heatmap[grid_cell] = {
                        'count': 0,
                        'made': 0,
                        'fg_pct': 0.0,
                        'avg_distance': 0.0,
                        'total_distance': 0.0
                    }
                
                heatmap[grid_cell]['count'] += 1
                heatmap[grid_cell]['made'] += int(shot['SHOT_MADE_FLAG'])
                heatmap[grid_cell]['total_distance'] += distance
            
            # Calculate percentages and averages
            for cell_data in heatmap.values():
                if cell_data['count'] > 0:
                    cell_data['fg_pct'] = cell_data['made'] / cell_data['count']
                    cell_data['avg_distance'] = cell_data['total_distance'] / cell_data['count']
            
            heatmaps[player_name] = heatmap
        
        return heatmaps
    
    def _calculate_usage_rates(self) -> Dict[str, float]:
        """
        Calculate usage rates for all players based on historical shot attempts.
        
        Usage rate represents the proportion of team shot attempts taken by
        this player. Usage rates for all players on a team sum to 1.0.
        
        Returns:
            Dictionary mapping player_name -> usage_rate (0.0 to 1.0)
        """
        usage_rates = {}
        
        # Calculate usage rate per team
        for team_name in self.shot_data['TEAM_NAME'].unique():
            team_shots = self.shot_data[self.shot_data['TEAM_NAME'] == team_name]
            total_team_shots = len(team_shots)
            
            if total_team_shots == 0:
                continue
            
            # Count shots per player on this team
            for player_name in team_shots['PLAYER_NAME'].unique():
                player_shots = team_shots[team_shots['PLAYER_NAME'] == player_name]
                player_shot_count = len(player_shots)
                
                # Usage rate = player shots / total team shots
                usage_rates[player_name] = player_shot_count / total_team_shots
        
        return usage_rates
    
    def _calculate_minutes_per_game(self) -> Dict[str, float]:
            """
            Calculate minutes per game for all players from shot data.

            Minutes are normalized so that each team's total minutes sum to exactly 240
            (5 players × 48 minutes), ensuring realistic game simulation. Players are
            capped at 40 minutes max, with leftover minutes distributed evenly to other players.

            If GAME_ID is available, estimates minutes based on games played and
            shot attempts. Otherwise, uses a default estimation based on usage rate.

            Returns:
                Dictionary mapping player_name -> minutes_per_game
            """
            minutes_per_game = {}

            # Check if GAME_ID is available for accurate game counting
            if 'GAME_ID' in self.shot_data.columns:
                # Count unique games per player
                for player_name in self.shot_data['PLAYER_NAME'].unique():
                    player_data = self.shot_data[self.shot_data['PLAYER_NAME'] == player_name]
                    games_played = player_data['GAME_ID'].nunique()
                    total_shots = len(player_data)

                    if games_played == 0:
                        minutes_per_game[player_name] = 0.0
                        continue

                    # Estimate minutes based on shot attempts
                    # Average NBA player takes ~1 shot per 2-3 minutes of play
                    # Starters: ~32-36 minutes, Bench: ~15-25 minutes
                    shots_per_game = total_shots / games_played

                    # Rough estimation: 2.5 minutes per shot attempt
                    # Cap at 40 minutes (max realistic playing time)
                    estimated_minutes = min(shots_per_game * 2.5, 40.0)

                    # Minimum of 10 minutes for players with any shots
                    estimated_minutes = max(estimated_minutes, 10.0) if total_shots > 0 else 0.0

                    minutes_per_game[player_name] = estimated_minutes
            else:
                # Fallback: estimate based on shot attempts across all data
                for player_name in self.shot_data['PLAYER_NAME'].unique():
                    player_data = self.shot_data[self.shot_data['PLAYER_NAME'] == player_name]
                    total_shots = len(player_data)

                    # Assume ~82 games in a season (rough estimate)
                    estimated_games = 82
                    shots_per_game = total_shots / estimated_games if estimated_games > 0 else 0

                    # Same estimation as above
                    estimated_minutes = min(shots_per_game * 2.5, 40.0)
                    estimated_minutes = max(estimated_minutes, 10.0) if total_shots > 0 else 0.0

                    minutes_per_game[player_name] = estimated_minutes

            # Normalize minutes per team to sum to exactly 240 (5 players × 48 minutes)
            # Cap players at 40 minutes and distribute leftover to other players
            for team_name in ['Golden State Warriors', 'Cleveland Cavaliers']:
                team_players = [
                    player for player in minutes_per_game.keys()
                    if self.shot_data[self.shot_data['PLAYER_NAME'] == player]['TEAM_NAME'].iloc[0] == team_name
                ]

                if not team_players:
                    continue

                # Identify players who are already at or near 40 minutes (cap them at exactly 40)
                capped_players = []
                other_players = []

                for player in team_players:
                    if minutes_per_game[player] >= 40.0:
                        minutes_per_game[player] = 40.0
                        capped_players.append(player)
                    else:
                        other_players.append(player)

                # Calculate current total for this team
                team_total = sum(minutes_per_game[player] for player in team_players)

                if team_total == 0:
                    continue

                # Calculate how many minutes we need to add to reach 240
                target_total = 240.0
                minutes_to_distribute = target_total - team_total

                if minutes_to_distribute > 0 and other_players:
                    # Distribute leftover minutes evenly to non-capped players only
                    minutes_per_player = minutes_to_distribute / len(other_players)

                    for player in other_players:
                        minutes_per_game[player] = round(minutes_per_game[player] + minutes_per_player, 1)
                elif minutes_to_distribute < 0:
                    # If we're over 240, scale down non-capped players proportionally
                    if other_players:
                        other_total = sum(minutes_per_game[player] for player in other_players)
                        if other_total > 0:
                            reduction_needed = -minutes_to_distribute
                            for player in other_players:
                                reduction = (minutes_per_game[player] / other_total) * reduction_needed
                                minutes_per_game[player] = round(minutes_per_game[player] - reduction, 1)

            return minutes_per_game



    
    def get_player_shot_distribution(self, 
                                     player_name: str,
                                     three_pt_radius: float,
                                     baseline_width: float) -> Dict[str, float]:
        """
        Get shot type distribution for a player given court dimensions.
        
        Args:
            player_name: Name of the player
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            
        Returns:
            Dictionary with keys 'rim', 'midrange', 'corner3', 'above_break3'
            representing the proportion of shots from each zone (sums to 1.0)
            
        Raises:
            ValueError: If player not found or dimensions out of range
        """
        # Validate inputs
        if not (23.75 <= three_pt_radius <= 26.0):
            raise ValueError(f"3-point radius must be between 23.75 and 26.0 feet, got {three_pt_radius}")
        if not (50.0 <= baseline_width <= 55.0):
            raise ValueError(f"Baseline width must be between 50.0 and 55.0 feet, got {baseline_width}")
        
        if player_name not in self.player_heatmaps:
            raise ValueError(f"Player not found: {player_name}")
        
        heatmap = self.player_heatmaps[player_name]
        
        # Count shots in each zone
        zone_counts = {
            'rim': 0,
            'midrange': 0,
            'corner3': 0,
            'above_break3': 0
        }
        
        for (grid_x, grid_y), cell_data in heatmap.items():
            # Convert grid coordinates back to feet
            loc_x = grid_x * self.grid_size
            loc_y = grid_y * self.grid_size
            
            # Classify shot based on court dimensions
            shot_type = self._classify_shot(loc_x, loc_y, three_pt_radius, baseline_width)
            zone_counts[shot_type] += cell_data['count']
        
        # Convert to proportions
        total_shots = sum(zone_counts.values())
        if total_shots == 0:
            # Default distribution if no shots
            return {'rim': 0.25, 'midrange': 0.25, 'corner3': 0.25, 'above_break3': 0.25}
        
        distribution = {k: v / total_shots for k, v in zone_counts.items()}
        return distribution
    
    def _classify_shot(self, loc_x: float, loc_y: float, 
                      three_pt_radius: float, baseline_width: float) -> str:
        """
        Classify a shot location as rim, midrange, corner3, or above_break3.
        
        Args:
            loc_x: X coordinate in feet (horizontal)
            loc_y: Y coordinate in feet (vertical, 0 is baseline)
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
            
        Returns:
            Shot type: 'rim', 'midrange', 'corner3', 'above_break3'
        """
        # Calculate distance from basket
        distance = math.sqrt(loc_x**2 + loc_y**2)
        
        # Rim shots: within 8 feet
        if distance < 8.0:
            return 'rim'
        
        # Check if shot is beyond 3-point line
        is_three_pointer = self._is_three_pointer(loc_x, loc_y, three_pt_radius, baseline_width)
        
        if is_three_pointer:
            # Determine if corner 3 or above-break 3
            if self._is_corner_three(loc_x, loc_y, three_pt_radius, baseline_width):
                return 'corner3'
            else:
                return 'above_break3'
        else:
            # 2-pointer beyond rim range is midrange
            return 'midrange'
    
    def _is_three_pointer(self, loc_x: float, loc_y: float,
                         three_pt_radius: float, baseline_width: float) -> bool:
        """
        Determine if a shot location is beyond the 3-point line.
        
        Args:
            loc_x: X coordinate in feet
            loc_y: Y coordinate in feet
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
            
        Returns:
            True if shot is a 3-pointer, False otherwise
        """
        distance = math.sqrt(loc_x**2 + loc_y**2)
        hoop_to_baseline = 4.0
        
        # NBA corner 3 is 22 feet from basket (straight line along sideline)
        # This is shorter than the arc radius (23.75 feet)
        # When the arc radius changes, scale the corner 3 distance proportionally
        standard_corner3_distance = 22.0
        standard_arc_radius = 23.75
        
        # Scale corner 3 distance based on arc radius change
        corner3_distance = standard_corner3_distance * (three_pt_radius / standard_arc_radius)
        
        # Check if in corner 3 area (close to baseline)
        if loc_y < hoop_to_baseline + 3.0:  # Within corner area
            # In corner area - use corner 3 distance
            return distance >= corner3_distance
        else:
            # Above break - check if beyond arc radius
            return distance >= three_pt_radius
    
    def _is_corner_three(self, loc_x: float, loc_y: float,
                        three_pt_radius: float, baseline_width: float) -> bool:
        """
        Determine if a 3-point shot is a corner 3.
        
        Args:
            loc_x: X coordinate in feet
            loc_y: Y coordinate in feet
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
            
        Returns:
            True if shot is a corner 3, False otherwise
        """
        hoop_to_baseline = 4.0
        
        # Corner 3s are close to baseline (within corner area)
        # and beyond the sideline threshold
        half_baseline = baseline_width / 2.0
        
        # Check if shot is in corner area (close to baseline)
        if loc_y < hoop_to_baseline + 3.0:  # Within 3 feet of corner area
            # Check if shot is near sideline
            if abs(loc_x) > half_baseline - 5.0:  # Within 5 feet of sideline
                return True
        
        return False
    
    def _calculate_corner3_distance(self, three_pt_radius: float, 
                                   baseline_width: float) -> float:
        """
        Calculate the corner 3-point distance from basket.
        
        Args:
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
            
        Returns:
            Corner 3 distance in feet
        """
        hoop_to_baseline = 4.0
        
        # Corner 3 distance is the minimum of the arc radius and 
        # the distance along the baseline
        corner3_distance = min(
            three_pt_radius,
            math.sqrt(three_pt_radius**2 - hoop_to_baseline**2)
        )
        
        return corner3_distance
    
    def get_player_expected_points(self,
                                  player_name: str,
                                  three_pt_radius: float,
                                  baseline_width: float) -> float:
        """
        Calculate expected points per shot for a player given court dimensions.
        
        Args:
            player_name: Name of the player
            three_pt_radius: 3-point line radius in feet
            baseline_width: Baseline width in feet
            
        Returns:
            Expected points per shot
            
        Raises:
            ValueError: If player not found or dimensions out of range
        """
        if player_name not in self.player_heatmaps:
            raise ValueError(f"Player not found: {player_name}")
        
        heatmap = self.player_heatmaps[player_name]
        
        total_expected_points = 0.0
        total_shots = 0
        
        for (grid_x, grid_y), cell_data in heatmap.items():
            # Convert grid coordinates back to feet
            loc_x = grid_x * self.grid_size
            loc_y = grid_y * self.grid_size
            
            # Determine shot value based on court dimensions
            is_three = self._is_three_pointer(loc_x, loc_y, three_pt_radius, baseline_width)
            shot_value = 3.0 if is_three else 2.0
            
            # Calculate expected points from this zone
            expected_points = cell_data['count'] * cell_data['fg_pct'] * shot_value
            total_expected_points += expected_points
            total_shots += cell_data['count']
        
        if total_shots == 0:
            return 0.0
        
        return total_expected_points / total_shots
    
    def get_player_usage_rate(self, player_name: str) -> float:
        """
        Get player usage rate from historical shot attempts.
        
        Usage rate represents the proportion of team shot attempts taken by
        this player. Usage rates for all players on a team sum to 1.0.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Usage rate as a float between 0.0 and 1.0
            
        Raises:
            ValueError: If player not found
        """
        if player_name not in self.player_usage_rates:
            raise ValueError(f"Player not found: {player_name}")
        
        return self.player_usage_rates[player_name]
    
    def get_player_minutes(self, player_name: str) -> float:
        """
        Get player's estimated minutes per game.
        
        Minutes are estimated based on shot attempts and games played.
        If GAME_ID is available in the data, estimates are more accurate.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Estimated minutes per game as a float
            
        Raises:
            ValueError: If player not found
        """
        if player_name not in self.player_minutes:
            raise ValueError(f"Player not found: {player_name}")
        
        return self.player_minutes[player_name]
    
    def get_team_players(self, team_name: str) -> List[str]:
        """
        Get list of players for a team.
        
        Args:
            team_name: Team name ('Golden State Warriors' or 'Cleveland Cavaliers')
            
        Returns:
            List of player names
            
        Raises:
            ValueError: If team not found
        """
        valid_teams = ['Golden State Warriors', 'Cleveland Cavaliers']
        if team_name not in valid_teams:
            raise ValueError(f"Team must be one of {valid_teams}, got {team_name}")
        
        team_players = self.shot_data[
            self.shot_data['TEAM_NAME'] == team_name
        ]['PLAYER_NAME'].unique().tolist()
        
        return team_players
    
    def sample_shot_location(self, player_name: str) -> Tuple[float, float]:
        """
        Sample a shot location from a player's spatial heatmap.
        
        Uses the player's historical shot distribution to randomly select
        a location where they are likely to shoot from. More frequently
        shot-from locations have higher probability of being selected.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Tuple of (loc_x, loc_y) in feet representing the sampled shot location
            
        Raises:
            ValueError: If player not found
        """
        if player_name not in self.player_heatmaps:
            raise ValueError(f"Player not found: {player_name}")
        
        heatmap = self.player_heatmaps[player_name]
        
        if not heatmap:
            # No shot data for player, return default location (mid-range)
            return (0.0, 15.0)
        
        # Create probability distribution based on shot counts
        grid_cells = list(heatmap.keys())
        shot_counts = [heatmap[cell]['count'] for cell in grid_cells]
        total_shots = sum(shot_counts)
        
        if total_shots == 0:
            return (0.0, 15.0)
        
        # Normalize to probabilities
        probabilities = [count / total_shots for count in shot_counts]
        
        # Sample a grid cell
        selected_idx = np.random.choice(len(grid_cells), p=probabilities)
        selected_cell = grid_cells[selected_idx]
        
        # Convert grid coordinates to feet and add random offset within cell
        grid_x, grid_y = selected_cell
        loc_x = grid_x * self.grid_size + np.random.uniform(0, self.grid_size)
        loc_y = grid_y * self.grid_size + np.random.uniform(0, self.grid_size)
        
        return (loc_x, loc_y)
    
    def get_shot_value(self, loc_x: float, loc_y: float,
                      three_pt_radius: float, baseline_width: float) -> int:
        """
        Determine the point value of a shot based on location and court configuration.
        
        Args:
            loc_x: X coordinate in feet (horizontal)
            loc_y: Y coordinate in feet (vertical, 0 is baseline)
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            
        Returns:
            Shot value: 2 or 3 points
            
        Raises:
            ValueError: If court dimensions are out of valid range
        """
        # Validate inputs
        if not (23.75 <= three_pt_radius <= 26.0):
            raise ValueError(f"3-point radius must be between 23.75 and 26.0 feet, got {three_pt_radius}")
        if not (50.0 <= baseline_width <= 55.0):
            raise ValueError(f"Baseline width must be between 50.0 and 55.0 feet, got {baseline_width}")
        
        # Check if shot is beyond 3-point line
        is_three = self._is_three_pointer(loc_x, loc_y, three_pt_radius, baseline_width)
        
        return 3 if is_three else 2
    
    def query_shot_probability(self, player_name: str, loc_x: float, loc_y: float) -> float:
        """
        Query the probability that a player shoots from a specific location.
        
        Returns the normalized probability based on the player's historical
        shot distribution. Locations where the player shoots more frequently
        will have higher probabilities.
        
        Args:
            player_name: Name of the player
            loc_x: X coordinate in feet (horizontal)
            loc_y: Y coordinate in feet (vertical)
            
        Returns:
            Probability value (0.0 to 1.0) representing the likelihood of
            the player shooting from this location. Returns 0.0 if the player
            has never shot from this grid cell.
            
        Raises:
            ValueError: If player not found
        """
        if player_name not in self.player_heatmaps:
            raise ValueError(f"Player not found: {player_name}")
        
        heatmap = self.player_heatmaps[player_name]
        
        if not heatmap:
            return 0.0
        
        # Convert location to grid coordinates using floor for correct binning
        grid_x = math.floor(loc_x / self.grid_size)
        grid_y = math.floor(loc_y / self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Get shot count for this cell
        if grid_cell not in heatmap:
            return 0.0
        
        cell_count = heatmap[grid_cell]['count']
        
        # Calculate total shots for normalization
        total_shots = sum(cell['count'] for cell in heatmap.values())
        
        if total_shots == 0:
            return 0.0
        
        # Return normalized probability
        return cell_count / total_shots
    
    def calculate_corner3_availability(self, 
                                       baseline_width: float,
                                       three_pt_radius: float) -> float:
        """
        Calculate geometric availability of corner 3-point shots.
        
        Wider baselines increase corner 3 availability because there's more
        space in the corners for players to position themselves for shots.
        
        Args:
            baseline_width: Baseline width in feet (50.0-55.0)
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            
        Returns:
            Availability factor (0.0-1.0+) representing relative corner 3 availability
            compared to standard court dimensions. Values > 1.0 indicate increased
            availability compared to standard court.
            
        Raises:
            ValueError: If dimensions are out of valid range
        """
        # Validate inputs
        if not (23.75 <= three_pt_radius <= 26.0):
            raise ValueError(f"3-point radius must be between 23.75 and 26.0 feet, got {three_pt_radius}")
        if not (50.0 <= baseline_width <= 55.0):
            raise ValueError(f"Baseline width must be between 50.0 and 55.0 feet, got {baseline_width}")
        
        # NBA court geometry:
        # - Hoop is 4 feet from baseline
        # - Corner 3 is the shorter of: 3pt radius or distance along baseline
        hoop_to_baseline = 4.0
        
        # Calculate corner 3 distance from basket
        corner3_distance = min(
            three_pt_radius,
            math.sqrt(three_pt_radius**2 - hoop_to_baseline**2)
        )
        
        # Calculate available corner space on each side
        half_baseline = baseline_width / 2.0
        
        # Calculate the horizontal distance from center to where corner 3 line meets sideline
        if three_pt_radius > hoop_to_baseline:
            corner3_horizontal = math.sqrt(three_pt_radius**2 - hoop_to_baseline**2)
        else:
            corner3_horizontal = 0.0
        
        # Available corner space
        corner_space = max(0.0, half_baseline - corner3_horizontal)
        
        # Calculate standard court corner space for comparison
        standard_half_baseline = self.standard_baseline_width / 2.0
        standard_corner3_horizontal = math.sqrt(self.standard_3pt_radius**2 - hoop_to_baseline**2)
        standard_corner_space = max(0.0, standard_half_baseline - standard_corner3_horizontal)
        
        # Calculate availability as ratio to standard court
        if standard_corner_space > 0:
            availability = corner_space / standard_corner_space
        else:
            availability = 1.0
        
        # Ensure availability is at least 0.0
        availability = max(0.0, availability)
        
        return availability
    
    def adjust_for_court_dimensions(self, 
                                    base_distribution: Dict[str, float],
                                    three_pt_radius: float, 
                                    baseline_width: float) -> Dict[str, float]:
        """
        Adjust shot distribution based on court dimensions.
        
        This method is maintained for backward compatibility but is deprecated.
        Use get_player_shot_distribution() for player-specific distributions.
        
        Logic:
        - Larger 3pt radius → fewer 3pt attempts, more 2pt attempts
        - Wider baseline → more corner 3pt availability
        
        Args:
            base_distribution: Base shot type frequencies with keys:
                'rim', 'midrange', 'corner3', 'above_break3'
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            
        Returns:
            Adjusted shot distribution dictionary summing to 1.0
            
        Raises:
            ValueError: If base_distribution doesn't sum to approximately 1.0
            ValueError: If court dimensions are out of valid range
        """
        # Validate inputs
        if not (23.75 <= three_pt_radius <= 26.0):
            raise ValueError(f"3-point radius must be between 23.75 and 26.0 feet, got {three_pt_radius}")
        if not (50.0 <= baseline_width <= 55.0):
            raise ValueError(f"Baseline width must be between 50.0 and 55.0 feet, got {baseline_width}")
        
        total = sum(base_distribution.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Base distribution must sum to approximately 1.0, got {total}")
        
        # Required keys
        required_keys = {'rim', 'midrange', 'corner3', 'above_break3'}
        if not required_keys.issubset(base_distribution.keys()):
            missing = required_keys - base_distribution.keys()
            raise ValueError(f"Base distribution missing required keys: {missing}")
        
        # Copy base distribution
        adjusted = base_distribution.copy()
        
        # Calculate adjustment factors
        radius_diff = three_pt_radius - self.standard_3pt_radius
        three_pt_adjustment = 1.0 - (radius_diff * 0.05)
        
        corner3_availability = self.calculate_corner3_availability(baseline_width, three_pt_radius)
        baseline_diff = baseline_width - self.standard_baseline_width
        corner3_adjustment = 1.0 + (baseline_diff * 0.02)
        
        # Apply adjustments
        adjusted['corner3'] = base_distribution['corner3'] * three_pt_adjustment
        adjusted['above_break3'] = base_distribution['above_break3'] * three_pt_adjustment
        adjusted['corner3'] = adjusted['corner3'] * corner3_adjustment * corner3_availability
        
        # Redistribute
        three_pt_reduction = (base_distribution['corner3'] + base_distribution['above_break3']) - \
                            (adjusted['corner3'] + adjusted['above_break3'])
        
        two_pt_total = base_distribution['rim'] + base_distribution['midrange']
        if two_pt_total > 0:
            rim_share = base_distribution['rim'] / two_pt_total
            midrange_share = base_distribution['midrange'] / two_pt_total
            
            adjusted['rim'] = base_distribution['rim'] + (three_pt_reduction * rim_share)
            adjusted['midrange'] = base_distribution['midrange'] + (three_pt_reduction * midrange_share)
        else:
            adjusted['rim'] = base_distribution['rim'] + (three_pt_reduction * 0.5)
            adjusted['midrange'] = base_distribution['midrange'] + (three_pt_reduction * 0.5)
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted

