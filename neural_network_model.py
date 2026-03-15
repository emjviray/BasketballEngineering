"""
Neural Network Model for NBA Court Optimization.

This module implements a feedforward neural network that predicts combined game scores
based on court dimensions and team statistics.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class NeuralNetworkModel:
    """Neural network for predicting combined game scores"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = None):
        """
        Initialize neural network architecture
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
            
        Architecture:
            Input → Dense(128, ReLU) → Dropout(0.2) → 
            Dense(64, ReLU) → Dropout(0.2) → 
            Dense(32, ReLU) → Dense(1, Linear) → Output
        """
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the network
        self.model = self._build_network()
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture"""
        layers = []
        
        # Input layer
        prev_size = self.input_dim
        
        # Hidden layers with dropout
        for i, hidden_size in enumerate(self.hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(self.hidden_layers) - 1:  # No dropout after last hidden layer
                layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, 1))
        
        return nn.Sequential(*layers)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training targets (combined scores)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            patience: Early stopping patience in epochs (default: 10)
            
        Returns:
            Training history with loss and validation metrics
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_tensor)
                val_loss = self.criterion(val_predictions, y_val_tensor)
                self.history['val_loss'].append(val_loss.item())
            
            # Early stopping check
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss.item():.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                # Restore best model
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict combined scores for given court configurations
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted combined scores (n_samples,)
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with MAE, RMSE, and R² metrics
        """
        predictions = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def save_model(self, filepath: str):
        """Save model weights to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
