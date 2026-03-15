#!/usr/bin/env python3
"""
NBA Court Optimization System - Main Orchestration Script

This script orchestrates the full pipeline:
1. Load shot data and grid data
2. Generate training data using game simulations
3. Extract and normalize features
4. Train neural network model
5. Run optimization to find optimal court dimensions
6. Validate results with game simulations
7. Display and save results
"""

import sys
import os
import argparse
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.neural_network_model import NeuralNetworkModel
from src.optimizer import Optimizer
from src.validator import Validator
from src.game_simulator import GameSimulator
from src.player_stats import PlayerStats
from src.shot_distribution import ShotDistribution

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='NBA Court Optimization System - Find optimal court dimensions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data files
    parser.add_argument(
        '--shot-data',
        type=str,
        default='warriors_cavs_playoff_shots_MASTER_2014_2024.csv',
        help='Path to shot location data CSV file'
    )
    parser.add_argument(
        '--grid-data',
        type=str,
        default='final_nn_input_full_grid.csv',
        help='Path to grid training data CSV file'
    )
    
    # Training parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of training samples to generate via simulation'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for neural network'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for neural network training'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Early stopping patience (epochs without improvement)'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=20,
        help='Number of game simulations per configuration during training'
    )
    parser.add_argument(
        '--validation-sims',
        type=int,
        default=100,
        help='Number of simulations for final validation'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--target-score',
        type=float,
        default=200.0,
        help='Target combined score for optimization'
    )
    parser.add_argument(
        '--grid-resolution',
        type=float,
        default=0.25,
        help='Grid resolution for optimization search (in feet)'
    )
    parser.add_argument(
        '--three-pt-min',
        type=float,
        default=23.75,
        help='Minimum 3-point radius to search (feet)'
    )
    parser.add_argument(
        '--three-pt-max',
        type=float,
        default=26.0,
        help='Maximum 3-point radius to search (feet)'
    )
    parser.add_argument(
        '--baseline-min',
        type=float,
        default=50.0,
        help='Minimum baseline width to search (feet)'
    )
    parser.add_argument(
        '--baseline-max',
        type=float,
        default=55.0,
        help='Maximum baseline width to search (feet)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to disk'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def generate_training_data(grid_data, shot_data, n_samples, n_simulations, verbose=False):
    """
    Generate training data by running game simulations.
    
    FIX: Generate training data per court configuration (not by sampling random rows).
    Each training sample represents: court_config + matchup_features → simulated_score
    
    Args:
        grid_data: Full grid data with team statistics
        shot_data: Shot location data
        n_samples: Number of samples to generate
        n_simulations: Number of simulations per configuration
        verbose: Enable verbose output
    
    Returns:
        DataFrame with features and simulated combined scores
    """
    if verbose:
        print(f"  Generating {n_samples} training samples...")
        print(f"  Running {n_simulations} simulation(s) per configuration")
    
    # Initialize simulation components
    shot_distribution = ShotDistribution(shot_data, grid_size=2.0)
    player_stats = PlayerStats(shot_distribution, eppg_calculator=None)
    
    # Extract team statistics (Warriors vs Cavaliers matchup)
    warriors_data = grid_data[grid_data['team'].str.contains('Warriors', case=False, na=False)].iloc[0]
    cavaliers_data = grid_data[grid_data['team'].str.contains('Cavaliers', case=False, na=False)].iloc[0]
    
    warriors_stats = {
        'pace': warriors_data['pace'],
        'off_reb_rate': warriors_data['off_reb_rate'],
        'def_reb_rate': warriors_data['def_reb_rate'],
        'turnover_rate': warriors_data['turnover_rate'],
        'free_throw_rate': warriors_data['free_throw_rate'],
        'rim_attempt_share': warriors_data['rim_attempt_share'],
        'midrange_share': warriors_data['midrange_share'],
        'corner3_pa': warriors_data['corner3_pa'],
        'above_break3_pa': warriors_data['above_break3_pa'],
        'threepar': warriors_data['threepar'],
        'team_rim_rate': warriors_data['team_rim_rate'],
        'team_corner3_rate': warriors_data['team_corner3_rate'],
        'opp_3par_allowed': cavaliers_data['threepar'],
        'opp_rim_fg_allowed': cavaliers_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    cavaliers_stats = {
        'pace': cavaliers_data['pace'],
        'off_reb_rate': cavaliers_data['off_reb_rate'],
        'def_reb_rate': cavaliers_data['def_reb_rate'],
        'turnover_rate': cavaliers_data['turnover_rate'],
        'free_throw_rate': cavaliers_data['free_throw_rate'],
        'rim_attempt_share': cavaliers_data['rim_attempt_share'],
        'midrange_share': cavaliers_data['midrange_share'],
        'corner3_pa': cavaliers_data['corner3_pa'],
        'above_break3_pa': cavaliers_data['above_break3_pa'],
        'threepar': cavaliers_data['threepar'],
        'team_rim_rate': cavaliers_data['team_rim_rate'],
        'team_corner3_rate': cavaliers_data['team_corner3_rate'],
        'opp_3par_allowed': warriors_data['threepar'],
        'opp_rim_fg_allowed': warriors_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    game_simulator = GameSimulator(
        warriors_stats=warriors_stats,
        cavaliers_stats=cavaliers_stats,
        player_stats=player_stats
    )
    
    # FIX: Generate court configurations systematically (not by sampling random rows)
    # Create grid of court configurations
    three_pt_radii = np.arange(23.75, 26.01, 0.25)
    baseline_widths = np.arange(50.0, 55.01, 0.25)
    
    # Create all combinations
    court_configs = []
    for radius in three_pt_radii:
        for width in baseline_widths:
            court_configs.append({
                'r_3pt_radius': radius,
                'baseline_width': width,
                'team': 'Warriors2016',  # For feature extraction compatibility
                'pace': warriors_data['pace'],
                'off_reb_rate': warriors_data['off_reb_rate'],
                'def_reb_rate': warriors_data['def_reb_rate'],
                'turnover_rate': warriors_data['turnover_rate'],
                'free_throw_rate': warriors_data['free_throw_rate'],
                'rim_attempt_share': warriors_data['rim_attempt_share'],
                'midrange_share': warriors_data['midrange_share'],
                'corner3_pa': warriors_data['corner3_pa'],
                'above_break3_pa': warriors_data['above_break3_pa'],
                'threepar': warriors_data['threepar'],
                'team_rim_rate': warriors_data['team_rim_rate'],
                'team_corner3_rate': warriors_data['team_corner3_rate'],
                'opp_3par_allowed': cavaliers_data['threepar'],
                'opp_rim_fg_allowed': cavaliers_data['team_rim_rate']
            })
    
    if verbose:
        print(f"  Generated {len(court_configs)} court configurations")
    
    # Sample if we have more configs than requested
    if len(court_configs) > n_samples:
        sampled_indices = np.random.choice(len(court_configs), n_samples, replace=False)
        court_configs = [court_configs[i] for i in sampled_indices]
        if verbose:
            print(f"  Sampled {len(court_configs)} configurations")
    
    # Run simulations for each configuration
    training_data = []
    combined_scores = []
    
    for idx, config in enumerate(court_configs):
        # Simulate games with this court configuration
        results = game_simulator.simulate_game(
            three_pt_radius=config['r_3pt_radius'],
            baseline_width=config['baseline_width'],
            n_simulations=n_simulations
        )
        
        # Calculate average combined score
        avg_combined = np.mean([w + c for w, c in results])
        combined_scores.append(avg_combined)
        
        # Store configuration with simulated score
        config['combined_score'] = avg_combined
        training_data.append(config)
        
        if verbose and ((idx + 1) % 100) == 0:
            print(f"    Progress: {idx + 1}/{len(court_configs)} configurations")
    
    training_df = pd.DataFrame(training_data)
    
    if verbose:
        print(f"  ✓ Generated {len(training_df)} training samples")
        print()
        print("  Training label statistics:")
        print(f"    Min:  {min(combined_scores):.1f}")
        print(f"    Max:  {max(combined_scores):.1f}")
        print(f"    Mean: {np.mean(combined_scores):.1f}")
        print(f"    Std:  {np.std(combined_scores):.1f}")
    
    return training_df


def save_results(result, validation_result, metrics, output_dir, args):
    """Save optimization results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main results as JSON
    results_dict = {
        'optimal_configuration': {
            'three_pt_radius': float(result['optimal_3pt_radius']),
            'baseline_width': float(result['optimal_baseline_width'])
        },
        'predicted_score': float(result['predicted_score']),
        'target_score': float(args.target_score),
        'score_difference': float(result['score_difference']),
        'validation': {
            'mean_simulated_score': float(validation_result['mean_simulated_score']),
            'std_simulated_score': float(validation_result['std_simulated_score']),
            'confidence_interval_95': [
                float(validation_result['confidence_interval_95'][0]),
                float(validation_result['confidence_interval_95'][1])
            ],
            'mae': float(validation_result['mae'])
        },
        'model_performance': {
            'test_mae': float(metrics['mae']),
            'test_rmse': float(metrics['rmse']),
            'test_r2': float(metrics['r2'])
        },
        'top_5_configurations': [
            {
                'rank': i + 1,
                'three_pt_radius': float(config.three_pt_radius),
                'baseline_width': float(config.baseline_width),
                'predicted_score': float(score),
                'difference_from_target': float(abs(score - args.target_score))
            }
            for i, (config, score) in enumerate(result['top_5_configs'])
        ]
    }
    
    results_file = output_path / 'optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"  ✓ Results saved to {results_file}")
    
    # Save detailed report as text
    report_file = output_path / 'optimization_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NBA COURT OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OPTIMAL CONFIGURATION:\n")
        f.write(f"  3-Point Radius:    {result['optimal_3pt_radius']:.2f} feet\n")
        f.write(f"  Baseline Width:    {result['optimal_baseline_width']:.2f} feet\n\n")
        
        f.write(f"  Predicted Score:   {result['predicted_score']:.2f} points\n")
        f.write(f"  Target Score:      {args.target_score:.2f} points\n")
        f.write(f"  Difference:        {result['score_difference']:.2f} points\n\n")
        
        f.write("VALIDATION RESULTS:\n")
        f.write(f"  Mean Simulated Score:  {validation_result['mean_simulated_score']:.2f} points\n")
        f.write(f"  Std Deviation:         {validation_result['std_simulated_score']:.2f} points\n")
        f.write(f"  95% CI: ({validation_result['confidence_interval_95'][0]:.2f}, "
                f"{validation_result['confidence_interval_95'][1]:.2f})\n")
        f.write(f"  Mean Absolute Error:   {validation_result['mae']:.2f} points\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"  Test MAE:  {metrics['mae']:.2f} points\n")
        f.write(f"  Test RMSE: {metrics['rmse']:.2f} points\n")
        f.write(f"  Test R²:   {metrics['r2']:.4f}\n\n")
        
        f.write("TOP 5 CONFIGURATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'3PT Radius':<15} {'Baseline Width':<18} {'Score':<12} {'Diff from Target':<15}\n")
        f.write("-" * 80 + "\n")
        
        for i, (config, score) in enumerate(result['top_5_configs'], 1):
            diff = abs(score - args.target_score)
            f.write(f"{i:<6} {config.three_pt_radius:<15.2f} {config.baseline_width:<18.2f} "
                   f"{score:<12.2f} {diff:<15.2f}\n")
    
    print(f"  ✓ Report saved to {report_file}")


def main():
    """Main orchestration function."""
    args = parse_arguments()
    
    print("=" * 80)
    print("NBA COURT OPTIMIZATION SYSTEM")
    print("=" * 80)
    print()
    
    # Display configuration
    if args.verbose:
        print("Configuration:")
        print(f"  Shot data: {args.shot_data}")
        print(f"  Grid data: {args.grid_data}")
        print(f"  Training samples: {args.n_samples}")
        print(f"  Simulations per sample: {args.n_simulations}")
        print(f"  Training epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Target score: {args.target_score}")
        print(f"  Grid resolution: {args.grid_resolution}")
        print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    print("-" * 80)
    start_time = time.time()
    
    data_loader = DataLoader()
    shot_data = data_loader.load_shot_data(args.shot_data)
    grid_data = data_loader.load_grid_data(args.grid_data)
    
    load_time = time.time() - start_time
    print(f"✓ Loaded {len(shot_data):,} shot records")
    print(f"✓ Loaded {len(grid_data):,} grid configurations")
    print(f"  Time: {load_time:.2f} seconds")
    print()
    
    # Step 2: Generate training data
    print("Step 2: Generating training data via game simulations...")
    print("-" * 80)
    start_time = time.time()
    
    # Extract team statistics for use in training and optimization
    warriors_data = grid_data[grid_data['team'].str.contains('Warriors', case=False, na=False)].iloc[0]
    cavaliers_data = grid_data[grid_data['team'].str.contains('Cavaliers', case=False, na=False)].iloc[0]
    
    warriors_stats = {
        'pace': warriors_data['pace'],
        'off_reb_rate': warriors_data['off_reb_rate'],
        'def_reb_rate': warriors_data['def_reb_rate'],
        'turnover_rate': warriors_data['turnover_rate'],
        'free_throw_rate': warriors_data['free_throw_rate'],
        'rim_attempt_share': warriors_data['rim_attempt_share'],
        'midrange_share': warriors_data['midrange_share'],
        'corner3_pa': warriors_data['corner3_pa'],
        'above_break3_pa': warriors_data['above_break3_pa'],
        'threepar': warriors_data['threepar'],
        'team_rim_rate': warriors_data['team_rim_rate'],
        'team_corner3_rate': warriors_data['team_corner3_rate'],
        'opp_3par_allowed': cavaliers_data['threepar'],
        'opp_rim_fg_allowed': cavaliers_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    cavaliers_stats = {
        'pace': cavaliers_data['pace'],
        'off_reb_rate': cavaliers_data['off_reb_rate'],
        'def_reb_rate': cavaliers_data['def_reb_rate'],
        'turnover_rate': cavaliers_data['turnover_rate'],
        'free_throw_rate': cavaliers_data['free_throw_rate'],
        'rim_attempt_share': cavaliers_data['rim_attempt_share'],
        'midrange_share': cavaliers_data['midrange_share'],
        'corner3_pa': cavaliers_data['corner3_pa'],
        'above_break3_pa': cavaliers_data['above_break3_pa'],
        'threepar': cavaliers_data['threepar'],
        'team_rim_rate': cavaliers_data['team_rim_rate'],
        'team_corner3_rate': cavaliers_data['team_corner3_rate'],
        'opp_3par_allowed': warriors_data['threepar'],
        'opp_rim_fg_allowed': warriors_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    training_grid = generate_training_data(
        grid_data, shot_data, 
        n_samples=args.n_samples,
        n_simulations=args.n_simulations,
        verbose=args.verbose
    )
    
    sim_time = time.time() - start_time
    print(f"✓ Generated {len(training_grid)} training samples")
    print(f"  Time: {sim_time:.2f} seconds ({sim_time/60:.2f} minutes)")
    print()
    
    # Step 3: Extract features
    print("Step 3: Extracting and normalizing features...")
    print("-" * 80)
    start_time = time.time()
    
    feature_eng = FeatureEngineering()
    features = feature_eng.extract_features(training_grid)
    normalized_features = feature_eng.normalize_features(features, fit=True)
    y = training_grid['combined_score'].values
    
    feature_time = time.time() - start_time
    print(f"✓ Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print(f"  Time: {feature_time:.2f} seconds")
    print()
    
    # Step 4: Split data
    print("Step 4: Splitting data...")
    print("-" * 80)
    
    n_samples = len(normalized_features)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = normalized_features[:train_size]
    y_train = y[:train_size]
    X_val = normalized_features[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = normalized_features[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print()
    
    # Step 5: Train neural network
    print("Step 5: Training neural network model...")
    print("-" * 80)
    start_time = time.time()
    
    model = NeuralNetworkModel(input_dim=normalized_features.shape[1])
    
    if args.verbose:
        print(f"  Architecture: {normalized_features.shape[1]} -> {' -> '.join(map(str, model.hidden_layers))} -> 1")
        print(f"  Device: {model.device}")
        print()
    
    print("  Training in progress...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    train_time = time.time() - start_time
    print()
    print(f"✓ Training completed")
    print(f"  Total epochs: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"  Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    print()
    
    # Step 6: Evaluate model
    print("Step 6: Evaluating model...")
    print("-" * 80)
    
    metrics = model.evaluate(X_test, y_test)
    print(f"✓ Test set performance:")
    print(f"  MAE:  {metrics['mae']:.2f} points")
    print(f"  RMSE: {metrics['rmse']:.2f} points")
    print(f"  R²:   {metrics['r2']:.4f}")
    
    # FIX: Add diagnostic output - inspect predictions vs actual values
    if args.verbose:
        print()
        print("  Sample predictions vs actual values:")
        print(f"  {'Predicted':<12} {'Actual':<12} {'Error':<12}")
        print("  " + "-" * 36)
        
        predictions = model.predict(X_test)
        for i in range(min(10, len(predictions))):
            error = abs(predictions[i] - y_test[i])
            print(f"  {predictions[i]:<12.2f} {y_test[i]:<12.2f} {error:<12.2f}")
    
    print()
    
    # Step 7: Run optimization
    print("Step 7: Running optimization...")
    print("-" * 80)
    start_time = time.time()
    
    # FIX: Pass team statistics to optimizer for accurate feature construction
    optimizer = Optimizer(model, feature_eng, warriors_stats=warriors_stats, cavaliers_stats=cavaliers_stats)
    
    if args.verbose:
        print(f"  Target score: {args.target_score} points")
        print(f"  3-point radius: [{args.three_pt_min}, {args.three_pt_max}] feet")
        print(f"  Baseline width: [{args.baseline_min}, {args.baseline_max}] feet")
        print(f"  Grid resolution: {args.grid_resolution} feet")
        print()
    
    result = optimizer.optimize(
        target_score=args.target_score,
        three_pt_range=(args.three_pt_min, args.three_pt_max),
        baseline_range=(args.baseline_min, args.baseline_max),
        grid_resolution=args.grid_resolution
    )
    
    opt_time = time.time() - start_time
    print(f"✓ Optimization completed")
    print(f"  Time: {opt_time:.2f} seconds")
    print()
    
    # Step 8: Display results
    print("Step 8: Optimization Results")
    print("=" * 80)
    print()
    print("OPTIMAL CONFIGURATION:")
    print(f"  3-Point Radius:    {result['optimal_3pt_radius']:.2f} feet")
    print(f"  Baseline Width:    {result['optimal_baseline_width']:.2f} feet")
    print()
    print(f"  Predicted Score:   {result['predicted_score']:.2f} points")
    print(f"  Target Score:      {args.target_score:.2f} points")
    print(f"  Difference:        {result['score_difference']:.2f} points")
    print()
    
    print("TOP 5 CONFIGURATIONS:")
    print("-" * 80)
    print(f"{'Rank':<6} {'3PT Radius':<15} {'Baseline Width':<18} {'Score':<12} {'Diff':<10}")
    print("-" * 80)
    
    for i, (config, score) in enumerate(result['top_5_configs'], 1):
        diff = abs(score - args.target_score)
        print(f"{i:<6} {config.three_pt_radius:<15.2f} {config.baseline_width:<18.2f} "
              f"{score:<12.2f} {diff:<10.2f}")
    print()
    
    # Step 9: Validate results
    print("Step 9: Validating optimal configuration...")
    print("-" * 80)
    start_time = time.time()
    
    # Reinitialize simulator for validation
    shot_distribution = ShotDistribution(shot_data, grid_size=2.0)
    player_stats = PlayerStats(shot_distribution, eppg_calculator=None)
    
    warriors_data = grid_data[grid_data['team'].str.contains('Warriors', case=False, na=False)].iloc[0]
    cavaliers_data = grid_data[grid_data['team'].str.contains('Cavaliers', case=False, na=False)].iloc[0]
    
    warriors_stats = {
        'pace': warriors_data['pace'],
        'off_reb_rate': warriors_data['off_reb_rate'],
        'def_reb_rate': warriors_data['def_reb_rate'],
        'turnover_rate': warriors_data['turnover_rate'],
        'free_throw_rate': warriors_data['free_throw_rate'],
        'rim_attempt_share': warriors_data['rim_attempt_share'],
        'midrange_share': warriors_data['midrange_share'],
        'corner3_pa': warriors_data['corner3_pa'],
        'above_break3_pa': warriors_data['above_break3_pa'],
        'threepar': warriors_data['threepar'],
        'team_rim_rate': warriors_data['team_rim_rate'],
        'team_corner3_rate': warriors_data['team_corner3_rate'],
        'opp_3par_allowed': cavaliers_data['threepar'],
        'opp_rim_fg_allowed': cavaliers_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    cavaliers_stats = {
        'pace': cavaliers_data['pace'],
        'off_reb_rate': cavaliers_data['off_reb_rate'],
        'def_reb_rate': cavaliers_data['def_reb_rate'],
        'turnover_rate': cavaliers_data['turnover_rate'],
        'free_throw_rate': cavaliers_data['free_throw_rate'],
        'rim_attempt_share': cavaliers_data['rim_attempt_share'],
        'midrange_share': cavaliers_data['midrange_share'],
        'corner3_pa': cavaliers_data['corner3_pa'],
        'above_break3_pa': cavaliers_data['above_break3_pa'],
        'threepar': cavaliers_data['threepar'],
        'team_rim_rate': cavaliers_data['team_rim_rate'],
        'team_corner3_rate': cavaliers_data['team_corner3_rate'],
        'opp_3par_allowed': warriors_data['threepar'],
        'opp_rim_fg_allowed': warriors_data['team_rim_rate'],
        'switch_rate': 0.3
    }
    
    game_simulator = GameSimulator(
        warriors_stats=warriors_stats,
        cavaliers_stats=cavaliers_stats,
        player_stats=player_stats
    )
    
    validator = Validator(game_simulator)
    
    print(f"  Running {args.validation_sims} game simulations...")
    
    validation_result = validator.validate_prediction(
        three_pt_radius=result['optimal_3pt_radius'],
        baseline_width=result['optimal_baseline_width'],
        predicted_score=result['predicted_score'],
        n_simulations=args.validation_sims
    )
    
    val_time = time.time() - start_time
    print(f"✓ Validation completed")
    print(f"  Time: {val_time:.2f} seconds")
    print()
    
    print("VALIDATION RESULTS:")
    print(f"  Mean Simulated Score:  {validation_result['mean_simulated_score']:.2f} points")
    print(f"  Std Deviation:         {validation_result['std_simulated_score']:.2f} points")
    print(f"  95% CI: ({validation_result['confidence_interval_95'][0]:.2f}, "
          f"{validation_result['confidence_interval_95'][1]:.2f})")
    print(f"  Mean Absolute Error:   {validation_result['mae']:.2f} points")
    print()
    
    # Check if prediction is within confidence interval
    ci_lower, ci_upper = validation_result['confidence_interval_95']
    if ci_lower <= result['predicted_score'] <= ci_upper:
        print(f"  ✓ Prediction is within 95% confidence interval")
    else:
        print(f"  ⚠ Prediction is outside 95% confidence interval")
    print()
    
    # Step 10: Save results
    print("Step 10: Saving results...")
    print("-" * 80)
    
    save_results(result, validation_result, metrics, args.output_dir, args)
    
    if args.save_model:
        model_path = Path(args.output_dir) / 'trained_model.pth'
        model.save_model(str(model_path))
        print(f"  ✓ Model saved to {model_path}")
    
    print()
    
    # Summary
    print("=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print()
    
    total_time = load_time + sim_time + feature_time + train_time + opt_time + val_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print()
    print("Pipeline stages:")
    print(f"  1. Data loading:       {load_time:.2f}s")
    print(f"  2. Training data gen:  {sim_time:.2f}s")
    print(f"  3. Feature extraction: {feature_time:.2f}s")
    print(f"  4. Model training:     {train_time:.2f}s")
    print(f"  5. Optimization:       {opt_time:.2f}s")
    print(f"  6. Validation:         {val_time:.2f}s")
    print()
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
