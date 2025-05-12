#!/usr/bin/env python3
"""
Statistical Strategies for NQ Alpha Elite

This module provides statistical and quantitative strategies for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

class MeanReversionZScoreStrategy(BaseStrategy):
    """
    Mean Reversion Z-Score Strategy
    
    This strategy uses z-scores to identify overbought and oversold conditions for mean reversion trading.
    """
    
    category = 'statistical'
    
    def __init__(self, lookback=20, entry_threshold=2.0, exit_threshold=0.5, logger=None):
        """
        Initialize the strategy
        
        Args:
            lookback (int): Lookback period for z-score calculation
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MeanReversionZScoreStrategy",
            description=f"Mean Reversion Z-Score Strategy (Lookback: {lookback}, Threshold: {entry_threshold})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'lookback': lookback,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
    
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
            
            # Get parameters
            lookback = self.parameters['lookback']
            entry_threshold = self.parameters['entry_threshold']
            exit_threshold = self.parameters['exit_threshold']
            
            # Calculate price changes
            df['Price_Change'] = df['Close'].pct_change()
            
            # Calculate rolling mean and standard deviation
            df['Rolling_Mean'] = df['Price_Change'].rolling(window=lookback).mean()
            df['Rolling_Std'] = df['Price_Change'].rolling(window=lookback).std()
            
            # Calculate z-score
            df['Z_Score'] = (df['Price_Change'] - df['Rolling_Mean']) / df['Rolling_Std']
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            df['Position'] = 0  # Track position
            
            # Entry signals
            # Buy signal: Z-score below negative threshold
            df.loc[df['Z_Score'] < -entry_threshold, 'Signal'] = 1
            
            # Sell signal: Z-score above positive threshold
            df.loc[df['Z_Score'] > entry_threshold, 'Signal'] = -1
            
            # Exit signals
            # Exit long: Z-score above negative exit threshold
            df.loc[(df['Z_Score'] > -exit_threshold) & (df['Position'] > 0), 'Signal'] = 0
            
            # Exit short: Z-score below positive exit threshold
            df.loc[(df['Z_Score'] < exit_threshold) & (df['Position'] < 0), 'Signal'] = 0
            
            # Track position
            position = 0
            for i in range(len(df)):
                if df['Signal'].iloc[i] == 1:
                    position = 1
                elif df['Signal'].iloc[i] == -1:
                    position = -1
                elif df['Signal'].iloc[i] == 0:
                    position = 0
                
                df.loc[df.index[i], 'Position'] = position
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy
    
    This strategy uses cointegration to identify pairs trading opportunities.
    """
    
    category = 'statistical'
    
    def __init__(self, lookback=60, entry_threshold=2.0, exit_threshold=0.5, logger=None):
        """
        Initialize the strategy
        
        Args:
            lookback (int): Lookback period for spread calculation
            entry_threshold (float): Spread threshold for entry
            exit_threshold (float): Spread threshold for exit
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="StatisticalArbitrageStrategy",
            description=f"Statistical Arbitrage Strategy (Lookback: {lookback}, Threshold: {entry_threshold})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'lookback': lookback,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
    
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
            
            # Get parameters
            lookback = self.parameters['lookback']
            entry_threshold = self.parameters['entry_threshold']
            exit_threshold = self.parameters['exit_threshold']
            
            # Calculate spread using synthetic pair
            # For NQ futures, we can use a synthetic pair with S&P 500 futures
            # Here we simulate it with a modified version of the price
            df['Synthetic_Pair'] = df['Close'] * (1 + np.random.normal(0, 0.001, len(df)))
            
            # Calculate hedge ratio using rolling regression
            df['Hedge_Ratio'] = 0.0
            
            for i in range(lookback, len(df)):
                # Get data for regression
                y = df['Close'].iloc[i-lookback:i].values
                x = df['Synthetic_Pair'].iloc[i-lookback:i].values
                x = sm.add_constant(x)
                
                # Run regression
                try:
                    model = sm.OLS(y, x).fit()
                    df.loc[df.index[i], 'Hedge_Ratio'] = model.params[1]
                except:
                    df.loc[df.index[i], 'Hedge_Ratio'] = df['Hedge_Ratio'].iloc[i-1]
            
            # Calculate spread
            df['Spread'] = df['Close'] - df['Hedge_Ratio'] * df['Synthetic_Pair']
            
            # Calculate z-score of spread
            df['Spread_Mean'] = df['Spread'].rolling(window=lookback).mean()
            df['Spread_Std'] = df['Spread'].rolling(window=lookback).std()
            df['Spread_Z'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            df['Position'] = 0  # Track position
            
            # Entry signals
            # Buy signal: Spread z-score below negative threshold
            df.loc[df['Spread_Z'] < -entry_threshold, 'Signal'] = 1
            
            # Sell signal: Spread z-score above positive threshold
            df.loc[df['Spread_Z'] > entry_threshold, 'Signal'] = -1
            
            # Exit signals
            # Exit long: Spread z-score above negative exit threshold
            df.loc[(df['Spread_Z'] > -exit_threshold) & (df['Position'] > 0), 'Signal'] = 0
            
            # Exit short: Spread z-score below positive exit threshold
            df.loc[(df['Spread_Z'] < exit_threshold) & (df['Position'] < 0), 'Signal'] = 0
            
            # Track position
            position = 0
            for i in range(len(df)):
                if df['Signal'].iloc[i] == 1:
                    position = 1
                elif df['Signal'].iloc[i] == -1:
                    position = -1
                elif df['Signal'].iloc[i] == 0:
                    position = 0
                
                df.loc[df.index[i], 'Position'] = position
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class KalmanFilterStrategy(BaseStrategy):
    """
    Kalman Filter Strategy
    
    This strategy uses a Kalman filter to estimate the trend and generate signals.
    """
    
    category = 'statistical'
    
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, logger=None):
        """
        Initialize the strategy
        
        Args:
            process_variance (float): Process variance for Kalman filter
            measurement_variance (float): Measurement variance for Kalman filter
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="KalmanFilterStrategy",
            description=f"Kalman Filter Strategy (PV: {process_variance}, MV: {measurement_variance})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'process_variance': process_variance,
            'measurement_variance': measurement_variance
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
    
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
            
            # Get parameters
            process_variance = self.parameters['process_variance']
            measurement_variance = self.parameters['measurement_variance']
            
            # Apply Kalman filter
            df = self._apply_kalman_filter(df, process_variance, measurement_variance)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price above Kalman estimate and slope is positive
            df.loc[(df['Close'] > df['Kalman_Estimate']) & 
                   (df['Kalman_Slope'] > 0), 'Signal'] = 1
            
            # Sell signal: Price below Kalman estimate and slope is negative
            df.loc[(df['Close'] < df['Kalman_Estimate']) & 
                   (df['Kalman_Slope'] < 0), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _apply_kalman_filter(self, df, process_variance, measurement_variance):
        """
        Apply Kalman filter to price data
        
        Args:
            df (DataFrame): Market data
            process_variance (float): Process variance
            measurement_variance (float): Measurement variance
            
        Returns:
            DataFrame: Market data with Kalman filter estimates
        """
        try:
            # Initialize Kalman filter
            kalman_estimate = df['Close'].iloc[0]
            kalman_error = 1.0
            
            # Arrays to store results
            estimates = np.zeros(len(df))
            errors = np.zeros(len(df))
            
            # Apply Kalman filter
            for i in range(len(df)):
                # Prediction step
                kalman_error = kalman_error + process_variance
                
                # Update step
                kalman_gain = kalman_error / (kalman_error + measurement_variance)
                kalman_estimate = kalman_estimate + kalman_gain * (df['Close'].iloc[i] - kalman_estimate)
                kalman_error = (1 - kalman_gain) * kalman_error
                
                # Store results
                estimates[i] = kalman_estimate
                errors[i] = kalman_error
            
            # Add to dataframe
            df['Kalman_Estimate'] = estimates
            df['Kalman_Error'] = errors
            
            # Calculate slope
            df['Kalman_Slope'] = df['Kalman_Estimate'].diff()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying Kalman filter: {e}")
            return df


class SeasonalityStrategy(BaseStrategy):
    """
    Seasonality Strategy
    
    This strategy identifies seasonal patterns and generates signals accordingly.
    """
    
    category = 'statistical'
    
    def __init__(self, lookback=252, seasonal_period=20, logger=None):
        """
        Initialize the strategy
        
        Args:
            lookback (int): Lookback period for seasonality analysis
            seasonal_period (int): Seasonal period in days
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="SeasonalityStrategy",
            description=f"Seasonality Strategy (Lookback: {lookback}, Period: {seasonal_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'lookback': lookback,
            'seasonal_period': seasonal_period
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
    
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
            
            # Get parameters
            lookback = self.parameters['lookback']
            seasonal_period = self.parameters['seasonal_period']
            
            # Check if we have enough data
            if len(df) < lookback + seasonal_period:
                return df
            
            # Calculate day of week if index is datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df['Day_of_Week'] = df.index.dayofweek
                df['Month'] = df.index.month
            else:
                # Create synthetic day of week
                df['Day_of_Week'] = np.arange(len(df)) % 5  # 0-4 for weekdays
                df['Month'] = (np.arange(len(df)) // 21) % 12 + 1  # 1-12 for months
            
            # Calculate seasonal patterns
            df = self._calculate_seasonal_patterns(df)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Strong positive seasonality
            df.loc[df['Seasonal_Score'] > 0.7, 'Signal'] = 1
            
            # Sell signal: Strong negative seasonality
            df.loc[df['Seasonal_Score'] < -0.7, 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _calculate_seasonal_patterns(self, df):
        """
        Calculate seasonal patterns
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with seasonal patterns
        """
        try:
            # Initialize seasonal score
            df['Seasonal_Score'] = 0.0
            
            # Calculate day of week effect
            day_returns = {}
            for day in range(5):  # 0-4 for weekdays
                day_data = df[df['Day_of_Week'] == day]
                if len(day_data) > 0:
                    day_returns[day] = day_data['Close'].pct_change().mean()
            
            # Calculate month effect
            month_returns = {}
            for month in range(1, 13):  # 1-12 for months
                month_data = df[df['Month'] == month]
                if len(month_data) > 0:
                    month_returns[month] = month_data['Close'].pct_change().mean()
            
            # Calculate seasonal score
            for i in range(len(df)):
                day = df['Day_of_Week'].iloc[i]
                month = df['Month'].iloc[i]
                
                day_score = day_returns.get(day, 0) * 100  # Convert to percentage
                month_score = month_returns.get(month, 0) * 100  # Convert to percentage
                
                # Combine scores
                seasonal_score = (day_score + month_score) / 2
                
                # Normalize to [-1, 1]
                seasonal_score = max(min(seasonal_score, 1), -1)
                
                df.loc[df.index[i], 'Seasonal_Score'] = seasonal_score
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal patterns: {e}")
            return df
