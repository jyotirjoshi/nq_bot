#!/usr/bin/env python3
"""
Pattern Recognition Strategies for NQ Alpha Elite

This module provides pattern recognition strategies for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

class EngulfingPatternStrategy(BaseStrategy):
    """
    Engulfing Pattern Strategy
    
    This strategy identifies bullish and bearish engulfing patterns and generates signals accordingly.
    """
    
    category = 'pattern'
    
    def __init__(self, confirmation_period=3, logger=None):
        """
        Initialize the strategy
        
        Args:
            confirmation_period (int): Number of bars to confirm the pattern
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="EngulfingPatternStrategy",
            description=f"Engulfing Pattern Strategy (Confirmation: {confirmation_period} bars)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'confirmation_period': confirmation_period
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
            confirmation_period = self.parameters['confirmation_period']
            
            # Calculate trend direction using SMA
            df = self.indicators.sma(df, [50])
            df['Trend'] = np.where(df['Close'] > df['SMA_50'], 1, -1)
            
            # Calculate candle body size
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            df['Body_Size_Pct'] = df['Body_Size'] / df['Close'] * 100
            
            # Identify bullish engulfing pattern
            df['Bullish_Engulfing'] = (
                (df['Close'] > df['Open']) &  # Current candle is bullish
                (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous candle is bearish
                (df['Close'] > df['Open'].shift(1)) &  # Current close is higher than previous open
                (df['Open'] < df['Close'].shift(1)) &  # Current open is lower than previous close
                (df['Body_Size'] > df['Body_Size'].shift(1) * 1.2)  # Current body is larger
            )
            
            # Identify bearish engulfing pattern
            df['Bearish_Engulfing'] = (
                (df['Close'] < df['Open']) &  # Current candle is bearish
                (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous candle is bullish
                (df['Close'] < df['Open'].shift(1)) &  # Current close is lower than previous open
                (df['Open'] > df['Close'].shift(1)) &  # Current open is higher than previous close
                (df['Body_Size'] > df['Body_Size'].shift(1) * 1.2)  # Current body is larger
            )
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Bullish engulfing in downtrend
            df.loc[(df['Bullish_Engulfing']) & 
                   (df['Trend'].shift(confirmation_period) < 0), 'Signal'] = 1
            
            # Sell signal: Bearish engulfing in uptrend
            df.loc[(df['Bearish_Engulfing']) & 
                   (df['Trend'].shift(confirmation_period) > 0), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class DojiPatternStrategy(BaseStrategy):
    """
    Doji Pattern Strategy
    
    This strategy identifies doji patterns at key levels and generates signals accordingly.
    """
    
    category = 'pattern'
    
    def __init__(self, doji_threshold=0.1, ma_period=20, logger=None):
        """
        Initialize the strategy
        
        Args:
            doji_threshold (float): Maximum body size as percentage of range for doji
            ma_period (int): Moving average period for trend identification
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="DojiPatternStrategy",
            description=f"Doji Pattern Strategy (Threshold: {doji_threshold}%, MA: {ma_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'doji_threshold': doji_threshold,
            'ma_period': ma_period
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
            doji_threshold = self.parameters['doji_threshold']
            ma_period = self.parameters['ma_period']
            
            # Calculate moving average for trend
            df = self.indicators.ema(df, [ma_period])
            df['Trend'] = np.where(df['Close'] > df[f'EMA_{ma_period}'], 1, -1)
            
            # Calculate candle properties
            df['Range'] = df['High'] - df['Low']
            df['Body'] = abs(df['Close'] - df['Open'])
            df['Body_Pct'] = df['Body'] / df['Range'] * 100
            
            # Identify doji patterns
            df['Doji'] = df['Body_Pct'] < doji_threshold
            
            # Identify doji at support/resistance
            df['Doji_Support'] = (
                df['Doji'] & 
                (df['Low'] < df['Low'].rolling(window=10).min() * 1.01) &
                (df['Trend'] < 0)
            )
            
            df['Doji_Resistance'] = (
                df['Doji'] & 
                (df['High'] > df['High'].rolling(window=10).max() * 0.99) &
                (df['Trend'] > 0)
            )
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Doji at support
            df.loc[df['Doji_Support'], 'Signal'] = 1
            
            # Sell signal: Doji at resistance
            df.loc[df['Doji_Resistance'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class HammerPatternStrategy(BaseStrategy):
    """
    Hammer Pattern Strategy
    
    This strategy identifies hammer and shooting star patterns and generates signals accordingly.
    """
    
    category = 'pattern'
    
    def __init__(self, body_pct=30, shadow_multiplier=2.0, logger=None):
        """
        Initialize the strategy
        
        Args:
            body_pct (float): Maximum body size as percentage of range
            shadow_multiplier (float): Minimum shadow to body ratio
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="HammerPatternStrategy",
            description=f"Hammer Pattern Strategy (Body: {body_pct}%, Shadow: {shadow_multiplier}×)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'body_pct': body_pct,
            'shadow_multiplier': shadow_multiplier
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
            body_pct = self.parameters['body_pct']
            shadow_multiplier = self.parameters['shadow_multiplier']
            
            # Calculate trend
            df = self.indicators.ema(df, [50])
            df['Trend'] = np.where(df['Close'] > df['EMA_50'], 1, -1)
            
            # Calculate candle properties
            df['Range'] = df['High'] - df['Low']
            df['Body'] = abs(df['Close'] - df['Open'])
            df['Body_Pct'] = df['Body'] / df['Range'] * 100
            
            # Calculate shadows
            df['Upper_Shadow'] = df.apply(
                lambda x: x['High'] - max(x['Open'], x['Close']), axis=1
            )
            df['Lower_Shadow'] = df.apply(
                lambda x: min(x['Open'], x['Close']) - x['Low'], axis=1
            )
            
            # Identify hammer pattern (bullish)
            df['Hammer'] = (
                (df['Body_Pct'] < body_pct) &  # Small body
                (df['Lower_Shadow'] > df['Body'] * shadow_multiplier) &  # Long lower shadow
                (df['Upper_Shadow'] < df['Body'])  # Short upper shadow
            )
            
            # Identify shooting star pattern (bearish)
            df['Shooting_Star'] = (
                (df['Body_Pct'] < body_pct) &  # Small body
                (df['Upper_Shadow'] > df['Body'] * shadow_multiplier) &  # Long upper shadow
                (df['Lower_Shadow'] < df['Body'])  # Short lower shadow
            )
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Hammer in downtrend
            df.loc[(df['Hammer']) & (df['Trend'] < 0), 'Signal'] = 1
            
            # Sell signal: Shooting star in uptrend
            df.loc[(df['Shooting_Star']) & (df['Trend'] > 0), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
