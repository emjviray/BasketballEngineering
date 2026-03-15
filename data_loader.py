"""
DataLoader module for NBA Court Optimization system.

This module provides functionality to load and validate CSV data files
containing shot data and grid training data.
"""

import os
from typing import List
import pandas as pd


class DataLoader:
    """Loads and validates CSV data files for NBA court optimization."""
    
    def load_shot_data(self, filepath: str) -> pd.DataFrame:
        """
        Load player shot data from CSV.
        
        Extracts player names, team names, shot coordinates (LOC_X, LOC_Y),
        shot outcomes (SHOT_MADE_FLAG), and shot metadata from the
        warriors_cavs_playoff_shots_MASTER_2014_2024.csv file.
        
        Args:
            filepath: Path to warriors_cavs_playoff_shots_MASTER_2014_2024.csv
            
        Returns:
            DataFrame with columns including: PLAYER_NAME, TEAM_NAME, LOC_X, 
            LOC_Y, SHOT_MADE_FLAG, and other shot metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist with descriptive message
            ValueError: If file format is invalid with specific format issue
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Shot data file not found: {filepath}. "
                f"Please ensure the file exists in the specified location."
            )
        
        # Load CSV file
        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Shot data file is empty: {filepath}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse shot data file {filepath}: {str(e)}"
            )
        
        # Validate required columns
        required_columns = [
            'PLAYER_NAME', 'TEAM_NAME', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Shot data file is missing required columns: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )
        
        # Validate numeric columns
        numeric_columns = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isna().all():
                        raise ValueError(
                            f"Column '{col}' contains no valid numeric values"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse numeric column '{col}': {str(e)}"
                    )
        
        return df
    
    def load_grid_data(self, filepath: str) -> pd.DataFrame:
        """
        Load full grid training data from CSV.
        
        Extracts team identifiers, court dimensions, shot attempt distributions,
        and team statistics from the final_nn_input_full_grid.csv file.
        
        Args:
            filepath: Path to final_nn_input_full_grid.csv
            
        Returns:
            DataFrame with team stats, court dimensions, and shot distributions
            
        Raises:
            FileNotFoundError: If file doesn't exist with descriptive message
            ValueError: If file format is invalid with specific format issue
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Grid data file not found: {filepath}. "
                f"Please ensure the file exists in the specified location."
            )
        
        # Load CSV file
        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Grid data file is empty: {filepath}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse grid data file {filepath}: {str(e)}"
            )
        
        # Validate required columns
        required_columns = [
            'team', 'r_3pt_radius', 'baseline_width',
            'corner3_pa', 'above_break3_pa', 'rim_attempt_share', 'midrange_share',
            'pace', 'off_reb_rate', 'turnover_rate', 'def_reb_rate',
            'opp_3par_allowed', 'opp_rim_fg_allowed'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Grid data file is missing required columns: {missing_columns}. "
                f"Expected columns include: {required_columns}"
            )
        
        # Validate numeric columns
        numeric_columns = [
            'r_3pt_radius', 'baseline_width', 'corner3_pa', 'above_break3_pa',
            'rim_attempt_share', 'midrange_share', 'pace', 'off_reb_rate',
            'turnover_rate', 'def_reb_rate', 'opp_3par_allowed', 'opp_rim_fg_allowed'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isna().all():
                        raise ValueError(
                            f"Column '{col}' contains no valid numeric values"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse numeric column '{col}': {str(e)}"
                    )
        
        return df
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of column names that must be present
            
        Returns:
            True if all required columns are present
            
        Raises:
            ValueError: If any required columns are missing
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )
        
        return True
