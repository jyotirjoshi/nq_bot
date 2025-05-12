#!/usr/bin/env python3
"""
Machine Learning Strategies for NQ Alpha Elite

This module provides machine learning strategies for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

class RandomForestStrategy(BaseStrategy):
    """
    Random Forest Strategy
    
    This strategy uses a Random Forest classifier to predict price movements and generate trading signals.
    """
    
    category = 'machine_learning'
    
    def __init__(self, n_estimators=100, max_depth=5, lookback=10, prediction_horizon=5, logger=None):
        """
        Initialize the strategy
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            lookback (int): Number of bars to look back for features
            prediction_horizon (int): Number of bars to predict ahead
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="RandomForestStrategy",
            description=f"Random Forest Strategy (Trees: {n_estimators}, Depth: {max_depth})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'lookback': lookback,
            'prediction_horizon': prediction_horizon
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
        
        # Initialize model
        self.model = None
        self.model_trained = False
        self.feature_columns = []
        
        # Initialize model path
        self.model_path = os.path.join(config.MODELS_DIR, "random_forest_model.joblib")
        
        # Try to load existing model
        self._load_model()
    
    def generate_signals(self, market_data):
        """
        Generate trading signals
        
        Args:
            market_data (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with signals
        """
        try:
            # Make a copy of market data
            df = market_data.copy()
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Train model if not trained
            if not self.model_trained and len(df) > 100:
                self._train_model(df)
            
            # Generate predictions if model is trained
            if self.model_trained:
                df = self._generate_predictions(df)
            else:
                # Use simple moving average crossover as fallback
                df = self.indicators.sma(df, [9, 21])
                df['Signal'] = 0  # Default to hold
                df.loc[df['SMA_9'] > df['SMA_21'], 'Signal'] = 1
                df.loc[df['SMA_9'] < df['SMA_21'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _add_indicators(self, df):
        """
        Add technical indicators for features
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with indicators
        """
        try:
            # Add RSI
            df = self.indicators.rsi(df)
            
            # Add MACD
            df = self.indicators.macd(df)
            
            # Add Bollinger Bands
            df = self.indicators.bollinger_bands(df)
            
            # Add moving averages
            df = self.indicators.sma(df, [9, 21, 50, 200])
            
            # Add ATR
            df = self.indicators.atr(df)
            
            # Add price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_1'] = df['Price_Change'].shift(1)
            df['Price_Change_2'] = df['Price_Change'].shift(2)
            df['Price_Change_3'] = df['Price_Change'].shift(3)
            
            # Add volume changes
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Add day of week
            if isinstance(df.index, pd.DatetimeIndex):
                df['Day_of_Week'] = df.index.dayofweek
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            return df
    
    def _train_model(self, df):
        """
        Train the Random Forest model
        
        Args:
            df (DataFrame): Market data with indicators
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Get parameters
            lookback = self.parameters['lookback']
            prediction_horizon = self.parameters['prediction_horizon']
            n_estimators = self.parameters['n_estimators']
            max_depth = self.parameters['max_depth']
            
            # Prepare features
            feature_columns = [
                'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower',
                'SMA_9', 'SMA_21', 'SMA_50', 'SMA_200', 'ATR',
                'Price_Change', 'Price_Change_1', 'Price_Change_2', 'Price_Change_3',
                'Volume_Change'
            ]
            
            # Add day of week if available
            if 'Day_of_Week' in df.columns:
                feature_columns.append('Day_of_Week')
            
            # Store feature columns
            self.feature_columns = feature_columns
            
            # Create target: 1 if price goes up in prediction_horizon bars, -1 if down
            df['Target'] = np.where(
                df['Close'].shift(-prediction_horizon) > df['Close'],
                1, -1
            )
            
            # Prepare data
            X = df[feature_columns].dropna()
            y = df['Target'].loc[X.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create and train model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            self.logger.info(f"Model trained: Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            self._save_model()
            
            # Set model as trained
            self.model_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
    
    def _generate_predictions(self, df):
        """
        Generate predictions using the trained model
        
        Args:
            df (DataFrame): Market data with indicators
            
        Returns:
            DataFrame: Market data with signals
        """
        try:
            # Check if model is trained
            if not self.model_trained:
                return df
            
            # Prepare features
            X = df[self.feature_columns].dropna()
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Add predictions to dataframe
            df.loc[X.index, 'ML_Prediction'] = predictions
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Model predicts price increase
            df.loc[df['ML_Prediction'] == 1, 'Signal'] = 1
            
            # Sell signal: Model predicts price decrease
            df.loc[df['ML_Prediction'] == -1, 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return df
    
    def _save_model(self):
        """Save the trained model"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load a previously trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.model_trained = True
                self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")


class GradientBoostingStrategy(BaseStrategy):
    """
    Gradient Boosting Strategy
    
    This strategy uses a Gradient Boosting classifier to predict price movements and generate trading signals.
    """
    
    category = 'machine_learning'
    
    def __init__(self, n_estimators=100, learning_rate=0.1, lookback=10, prediction_horizon=5, logger=None):
        """
        Initialize the strategy
        
        Args:
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate
            lookback (int): Number of bars to look back for features
            prediction_horizon (int): Number of bars to predict ahead
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="GradientBoostingStrategy",
            description=f"Gradient Boosting Strategy (Estimators: {n_estimators}, LR: {learning_rate})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'lookback': lookback,
            'prediction_horizon': prediction_horizon
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
        
        # Initialize model
        self.model = None
        self.model_trained = False
        self.feature_columns = []
        
        # Initialize model path
        self.model_path = os.path.join(config.MODELS_DIR, "gradient_boosting_model.joblib")
        
        # Try to load existing model
        self._load_model()
    
    # The rest of the implementation is similar to RandomForestStrategy
    # with the model changed to GradientBoostingClassifier
