"""
Game simulation engine for NBA Court Optimization system.

This module simulates basketball games between two teams using possession-based
logic, incorporating team statistics, player performance, and court dimensions.
"""

from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import math
from src.models import CourtConfiguration, GameResult
from src.player_stats import PlayerStats
from src.shot_distribution import ShotDistribution


class GameSimulator:
    """Simulates basketball games between two teams"""
    
    def __init__(self, 
                 warriors_stats: Dict, 
                 cavaliers_stats: Dict, 
                 player_stats: PlayerStats,
                 eppg_calculator=None,
                 shot_distribution=None):
        """
        Initialize simulator with team and player data.
        
        Args:
            warriors_stats: Team statistics for Warriors with keys:
                - pace: Possessions per game
                - off_reb_rate: Offensive rebound rate
                - def_reb_rate: Defensive rebound rate
                - turnover_rate: Turnover rate
                - free_throw_rate: Free throw attempts per FGA
                - rim_attempt_share: Proportion of shots at rim
                - midrange_share: Proportion of midrange shots
                - corner3_pa: Corner 3 attempts per game
                - above_break3_pa: Above-break 3 attempts per game
                - threepar: 3-point attempt rate
                - team_rim_rate: Rim FG%
                - team_corner3_rate: Corner 3 FG%
                - opp_3par_allowed: Opponent 3PA rate allowed
                - opp_rim_fg_allowed: Opponent rim FG% allowed
            cavaliers_stats: Team statistics for Cavaliers (same structure)
            player_stats: PlayerStats instance for accessing player data
            eppg_calculator: EPPGCalculator for dynamic EPPG calculation (optional)
            shot_distribution: ShotDistribution instance (optional, uses player_stats.shot_distribution if not provided)
        """
        self.warriors_stats = warriors_stats
        self.cavaliers_stats = cavaliers_stats
        self.player_stats = player_stats
        self.eppg_calculator = eppg_calculator
        
        # Use provided shot_distribution or get from player_stats
        if shot_distribution is not None:
            self.shot_distribution = shot_distribution
        elif hasattr(player_stats, 'shot_distribution'):
            self.shot_distribution = player_stats.shot_distribution
        else:
            # Fallback: create a basic shot distribution from player_stats shot data
            self.shot_distribution = None
        
        # Get team rosters
        self.warriors_players = player_stats.get_team_players('Golden State Warriors')
        self.cavaliers_players = player_stats.get_team_players('Cleveland Cavaliers')
    
    def simulate_game(self, 
                     three_pt_radius: float, 
                     baseline_width: float,
                     n_simulations: int = 50) -> List[Tuple[int, int]]:
        """
        Simulate game(s) with specified court dimensions.
        
        This method simulates complete games with realistic player minutes tracking,
        player rotation based on minutes played, and heatmap-based shot selection.
        
        Args:
            three_pt_radius: 3-point line radius in feet (23.75-26.0)
            baseline_width: Baseline width in feet (50.0-55.0)
            n_simulations: Number of games to simulate (default: 50)
            
        Returns:
            List of (warriors_score, cavaliers_score) tuples
        """
        results = []
        
        for _ in range(n_simulations):
            # Create court configuration
            court_config = {
                'three_pt_radius': three_pt_radius,
                'baseline_width': baseline_width
            }
            
            # Calculate total possessions based on pace
            # Average the two teams' pace values
            avg_pace = (self.warriors_stats['pace'] + self.cavaliers_stats['pace']) / 2.0
            # FIX: Multiply by 2 because possessions alternate between teams
            # Each team should get ~avg_pace possessions, so total is ~2*avg_pace
            total_possessions = int(avg_pace * 2)
            
            # Initialize scores and possession counts
            warriors_score = 0
            cavaliers_score = 0
            warriors_possessions = 0
            cavaliers_possessions = 0
            
            # Track shot breakdown
            warriors_shots = {'rim': 0, 'midrange': 0, 'corner3': 0, 'above_break3': 0}
            cavaliers_shots = {'rim': 0, 'midrange': 0, 'corner3': 0, 'above_break3': 0}
            
            # Initialize player minutes tracking for this game
            # Track minutes played so far in this game for each player
            warriors_minutes_played = {player: 0.0 for player in self.warriors_players}
            cavaliers_minutes_played = {player: 0.0 for player in self.cavaliers_players}
            
            # Get target minutes per game for each player
            warriors_target_minutes = {
                player: self.player_stats.get_player_minutes(player) 
                for player in self.warriors_players
            }
            cavaliers_target_minutes = {
                player: self.player_stats.get_player_minutes(player) 
                for player in self.cavaliers_players
            }
            
            # Estimate game duration in minutes (48 minutes regulation)
            game_duration = 48.0
            minutes_per_possession = game_duration / total_possessions if total_possessions > 0 else 0
            
            # Simulate possessions alternating between teams
            for i in range(total_possessions):
                # Alternate possessions (Warriors start)
                if i % 2 == 0:
                    offensive_team = 'warriors'
                    offensive_stats = self.warriors_stats
                    defensive_stats = self.cavaliers_stats
                    warriors_possessions += 1
                    team_players = self.warriors_players
                    minutes_played = warriors_minutes_played
                    target_minutes = warriors_target_minutes
                else:
                    offensive_team = 'cavaliers'
                    offensive_stats = self.cavaliers_stats
                    defensive_stats = self.warriors_stats
                    cavaliers_possessions += 1
                    team_players = self.cavaliers_players
                    minutes_played = cavaliers_minutes_played
                    target_minutes = cavaliers_target_minutes
                
                # Get active players (those who haven't exceeded their minutes)
                active_players = self._get_active_players(
                    team_players, minutes_played, target_minutes
                )
                
                # If no active players (shouldn't happen), use all players
                if not active_players:
                    active_players = team_players
                
                # Define the 5 players on court (top 5 by priority)
                on_court = active_players[:5]
                
                # Select which player takes the shot from those actually on court
                shooter = self._select_shooter(
                    on_court, offensive_team, court_config
                )
                
                # Simulate possession with the selected shooter
                points, shot_type = self.simulate_possession_with_player(
                    shooter,
                    offensive_team, 
                    offensive_stats,
                    defensive_stats,
                    court_config
                )
                
                # Update scores
                if offensive_team == 'warriors':
                    warriors_score += points
                    if shot_type:
                        warriors_shots[shot_type] += 1
                else:
                    cavaliers_score += points
                    if shot_type:
                        cavaliers_shots[shot_type] += 1
                
                # Update minutes played for players actually on court
                for player in on_court:
                    minutes_played[player] += minutes_per_possession
            
            results.append((warriors_score, cavaliers_score))
        
        return results
    
    def _get_active_players(self, 
                           team_players: List[str],
                           minutes_played: Dict[str, float],
                           target_minutes: Dict[str, float]) -> List[str]:
        """
        Get list of active players who haven't exceeded their target minutes.
        
        Players are prioritized by:
        1. Those who haven't reached their target minutes
        2. Sorted by usage rate (higher usage = higher priority)
        
        Args:
            team_players: List of all players on the team
            minutes_played: Dict mapping player -> minutes played so far
            target_minutes: Dict mapping player -> target minutes per game
            
        Returns:
            List of active player names, sorted by priority
        """
        # Get players who haven't exceeded their minutes
        available = [
            player for player in team_players
            if minutes_played[player] < target_minutes[player]
        ]
        
        # If no one is available (everyone exceeded minutes), use all players
        if not available:
            available = team_players
        
        # Sort by usage rate (higher usage = higher priority)
        available.sort(
            key=lambda p: self.player_stats.get_player_usage_rate(p),
            reverse=True
        )
        
        return available
    
    def _select_shooter(self,
                       active_players: List[str],
                       offensive_team: str,
                       court_config: Dict) -> str:
        """
        Select which player takes the shot based on usage rates.
        
        Players with higher usage rates have higher probability of being selected.
        Only considers active players (those currently on the court).
        
        Args:
            active_players: List of players currently on court
            offensive_team: 'warriors' or 'cavaliers'
            court_config: Court configuration dict
            
        Returns:
            Name of the player who takes the shot
        """
        if not active_players:
            # Fallback: shouldn't happen, but return first player if it does
            return self.warriors_players[0] if offensive_team == 'warriors' else self.cavaliers_players[0]
        
        # Get usage rates for active players
        usage_rates = [
            self.player_stats.get_player_usage_rate(player)
            for player in active_players
        ]
        
        # Normalize usage rates to sum to 1.0
        total_usage = sum(usage_rates)
        if total_usage > 0:
            probabilities = [rate / total_usage for rate in usage_rates]
        else:
            # Equal probability if no usage data
            probabilities = [1.0 / len(active_players)] * len(active_players)
        
        # Select shooter based on usage probabilities
        shooter = random.choices(active_players, weights=probabilities, k=1)[0]
        
        return shooter
    
    def simulate_possession_with_player(self,
                                       shooter: str,
                                       offensive_team: str,
                                       offensive_stats: Dict,
                                       defensive_stats: Dict,
                                       court_config: Dict) -> Tuple[int, Optional[str]]:
        """
        Simulate a single possession with a specific player taking the shot.
        
        Uses player-specific spatial heatmaps to determine shot location,
        then calculates shot value and success probability based on that location.
        
        Args:
            shooter: Name of the player taking the shot
            offensive_team: 'warriors' or 'cavaliers'
            offensive_stats: Offensive team statistics
            defensive_stats: Defensive team statistics
            court_config: Court configuration dict with three_pt_radius and baseline_width
            
        Returns:
            Tuple of (points_scored, shot_type) where shot_type is one of:
            'rim', 'midrange', 'corner3', 'above_break3', or None for turnover
        """
        # Check for turnover
        if random.random() < offensive_stats['turnover_rate']:
            return (0, None)
        
        # Sample shot location from player's spatial heatmap
        if self.shot_distribution is not None:
            loc_x, loc_y = self.shot_distribution.sample_shot_location(shooter)
            
            # Determine shot value (2PT or 3PT) based on location and court config
            shot_value = self.shot_distribution.get_shot_value(
                loc_x, loc_y,
                court_config['three_pt_radius'],
                court_config['baseline_width']
            )
            
            # Classify shot type for tracking
            shot_type = self.shot_distribution._classify_shot(
                loc_x, loc_y,
                court_config['three_pt_radius'],
                court_config['baseline_width']
            )
            
            # Get player's FG% at this location from heatmap
            shot_prob = self._get_player_shot_probability(shooter, loc_x, loc_y)
            
            # Apply defensive adjustments
            shot_prob = self._apply_defensive_adjustments(
                shot_prob, shot_type, offensive_stats, defensive_stats
            )
        else:
            # Fallback: use old team-based shot selection
            shot_type = self.select_shot_type(offensive_stats, court_config)
            shot_value = 3 if shot_type in ['corner3', 'above_break3'] else 2
            shot_prob = self.calculate_shot_probability(
                shot_type, offensive_stats, defensive_stats
            )
        
        # Determine if shot is made
        shot_made = random.random() < shot_prob
        
        if shot_made:
            points = shot_value
            
            # Check for free throws (and-one opportunity)
            if random.random() < offensive_stats['free_throw_rate'] * 0.2:  # 20% of FT rate for and-ones
                # Assume 75% FT shooting for simplicity
                if random.random() < 0.75:
                    points += 1
            
            return (points, shot_type)
        else:
            # Miss - check for offensive rebound
            if random.random() < offensive_stats['off_reb_rate']:
                # Offensive rebound - simulate another shot attempt (simplified)
                # Use same shot type, lower probability
                if random.random() < shot_prob * 0.8:
                    return (shot_value, shot_type)
            
            # Check for free throws on missed shot (shooting foul)
            if random.random() < offensive_stats['free_throw_rate'] * 0.3:  # 30% of FT rate
                # Shooting foul - 2 or 3 free throws
                num_fts = 3 if shot_value == 3 else 2
                ft_points = 0
                for _ in range(num_fts):
                    if random.random() < 0.75:  # 75% FT shooting
                        ft_points += 1
                return (ft_points, shot_type)
            
            return (0, shot_type)
    
    def _get_player_shot_probability(self, player_name: str, 
                                     loc_x: float, loc_y: float) -> float:
        """
        Get player's shooting probability at a specific location from heatmap.
        
        Args:
            player_name: Name of the player
            loc_x: X coordinate in feet
            loc_y: Y coordinate in feet
            
        Returns:
            Field goal percentage at this location (0.0-1.0)
        """
        if self.shot_distribution is None:
            return 0.45  # Default league average
        
        if player_name not in self.shot_distribution.player_heatmaps:
            return 0.45  # Default league average
        
        heatmap = self.shot_distribution.player_heatmaps[player_name]
        
        # Convert location to grid coordinates using floor for correct binning
        grid_x = math.floor(loc_x / self.shot_distribution.grid_size)
        grid_y = math.floor(loc_y / self.shot_distribution.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Get FG% from heatmap
        if grid_cell in heatmap:
            return heatmap[grid_cell]['fg_pct']
        else:
            # No data for this location, use league average
            return 0.45
    
    def _apply_defensive_adjustments(self,
                                    base_prob: float,
                                    shot_type: str,
                                    offensive_stats: Dict,
                                    defensive_stats: Dict) -> float:
        """
        Apply defensive adjustments to shot probability.
        
        Args:
            base_prob: Base shooting probability from player heatmap
            shot_type: Type of shot ('rim', 'midrange', 'corner3', 'above_break3')
            offensive_stats: Offensive team statistics
            defensive_stats: Defensive team statistics
            
        Returns:
            Adjusted probability (0.0-1.0)
        """
        adjusted_prob = base_prob
        
        # Apply defensive adjustments based on shot type
        if shot_type == 'rim':
            # Apply defensive rim FG% allowed
            if 'opp_rim_fg_allowed' in defensive_stats:
                defensive_factor = defensive_stats['opp_rim_fg_allowed'] / 0.65  # Normalize to league average
                adjusted_prob *= defensive_factor
        
        elif shot_type in ['corner3', 'above_break3']:
            # Apply defensive 3PA rate allowed
            if 'opp_3par_allowed' in defensive_stats:
                # Higher opp_3par_allowed means defense allows more 3s (worse defense)
                defensive_factor = defensive_stats['opp_3par_allowed'] / 0.35
                adjusted_prob *= min(defensive_factor, 1.15)  # Cap at 15% boost
        
        # Ensure probability is in valid range [0.0, 1.0]
        return max(0.0, min(1.0, adjusted_prob))
    
    def simulate_possession(self, 
                          offensive_team: str,
                          offensive_stats: Dict,
                          defensive_stats: Dict,
                          court_config: Dict) -> Tuple[int, Optional[str]]:
        """
        Simulate a single possession (legacy method for backward compatibility).
        
        This method is maintained for backward compatibility but now delegates
        to simulate_possession_with_player() with a randomly selected shooter.
        
        Args:
            offensive_team: 'warriors' or 'cavaliers'
            offensive_stats: Offensive team statistics
            defensive_stats: Defensive team statistics
            court_config: Court configuration dict with three_pt_radius and baseline_width
            
        Returns:
            Tuple of (points_scored, shot_type) where shot_type is one of:
            'rim', 'midrange', 'corner3', 'above_break3', or None for turnover
        """
        # Select a random shooter based on usage rates
        team_players = self.warriors_players if offensive_team == 'warriors' else self.cavaliers_players
        
        # Create dummy minutes tracking (all players available)
        minutes_played = {player: 0.0 for player in team_players}
        target_minutes = {player: 40.0 for player in team_players}
        
        active_players = self._get_active_players(team_players, minutes_played, target_minutes)
        shooter = self._select_shooter(active_players, offensive_team, court_config)
        
        # Delegate to new method
        return self.simulate_possession_with_player(
            shooter, offensive_team, offensive_stats, defensive_stats, court_config
        )
    
    def select_shot_type(self, team_stats: Dict, court_config: Dict) -> str:
        """
        Determine shot type based on team tendencies and court dimensions.
        
        Args:
            team_stats: Team statistics
            court_config: Court configuration with three_pt_radius and baseline_width
            
        Returns:
            Shot type: 'rim', 'midrange', 'corner3', 'above_break3'
        """
        # Build base distribution from team stats
        # Calculate total shot attempts to get proportions
        total_3pa = team_stats['corner3_pa'] + team_stats['above_break3_pa']
        total_2pa = 100.0 - total_3pa  # Assume 100 total attempts for proportion
        
        # Distribute 2PA between rim and midrange based on shares
        rim_share = team_stats['rim_attempt_share']
        midrange_share = team_stats['midrange_share']
        two_pt_total = rim_share + midrange_share
        
        if two_pt_total > 0:
            rim_proportion = (rim_share / two_pt_total) * (total_2pa / 100.0)
            midrange_proportion = (midrange_share / two_pt_total) * (total_2pa / 100.0)
        else:
            rim_proportion = 0.5
            midrange_proportion = 0.5
        
        corner3_proportion = team_stats['corner3_pa'] / 100.0
        above_break3_proportion = team_stats['above_break3_pa'] / 100.0
        
        # Create base distribution
        base_distribution = {
            'rim': rim_proportion,
            'midrange': midrange_proportion,
            'corner3': corner3_proportion,
            'above_break3': above_break3_proportion
        }
        
        # Normalize to sum to 1.0
        total = sum(base_distribution.values())
        if total > 0:
            base_distribution = {k: v / total for k, v in base_distribution.items()}
        else:
            # Fallback to even distribution
            base_distribution = {
                'rim': 0.25,
                'midrange': 0.25,
                'corner3': 0.25,
                'above_break3': 0.25
            }
        
        # Adjust distribution based on court dimensions if shot_distribution is available
        if self.shot_distribution is not None:
            adjusted_distribution = self.shot_distribution.adjust_for_court_dimensions(
                base_distribution,
                court_config['three_pt_radius'],
                court_config['baseline_width']
            )
        else:
            # Use base distribution without adjustment
            adjusted_distribution = base_distribution
        
        # Select shot type based on adjusted probabilities
        shot_types = list(adjusted_distribution.keys())
        probabilities = list(adjusted_distribution.values())
        
        return random.choices(shot_types, weights=probabilities, k=1)[0]
    
    def calculate_shot_probability(self, 
                                   shot_type: str, 
                                   offensive_stats: Dict,
                                   defensive_stats: Dict) -> float:
        """
        Calculate probability of shot success.
        
        Args:
            shot_type: 'rim', 'midrange', 'corner3', 'above_break3'
            offensive_stats: Offensive team statistics
            defensive_stats: Defensive team statistics
            
        Returns:
            Probability of shot success (0.0-1.0)
        """
        # Base shooting percentages by shot type (league averages)
        base_percentages = {
            'rim': 0.65,
            'midrange': 0.40,
            'corner3': 0.38,
            'above_break3': 0.35
        }
        
        # Get base probability
        base_prob = base_percentages.get(shot_type, 0.45)
        
        # Adjust based on team shooting efficiency
        if shot_type == 'rim':
            # Use team rim rate if available
            if 'team_rim_rate' in offensive_stats:
                base_prob = offensive_stats['team_rim_rate']
            
            # Apply defensive rim FG% allowed
            if 'opp_rim_fg_allowed' in defensive_stats:
                defensive_factor = defensive_stats['opp_rim_fg_allowed'] / 0.65  # Normalize to league average
                base_prob *= defensive_factor
        
        elif shot_type == 'corner3':
            # Use team corner 3 rate if available
            if 'team_corner3_rate' in offensive_stats:
                base_prob = offensive_stats['team_corner3_rate']
        
        elif shot_type in ['above_break3']:
            # Adjust based on 3-point attempt rate (higher 3PAR suggests better 3pt shooting)
            if 'threepar' in offensive_stats:
                three_pt_factor = offensive_stats['threepar'] / 0.35  # Normalize to league average
                base_prob *= min(three_pt_factor, 1.2)  # Cap at 20% boost
            
            # Apply defensive 3PA rate allowed
            if 'opp_3par_allowed' in defensive_stats:
                # Higher opp_3par_allowed means defense allows more 3s (worse defense)
                defensive_factor = defensive_stats['opp_3par_allowed'] / 0.35
                base_prob *= min(defensive_factor, 1.15)  # Cap at 15% boost
        
        # Ensure probability is in valid range [0.0, 1.0]
        return max(0.0, min(1.0, base_prob))
