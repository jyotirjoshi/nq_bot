#!/usr/bin/env python3
"""
Volatility-Based Strategies for NQ Alpha Elite

This module provides volatility-based strategies for trading NASDAQ 100 E-mini futures.
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

class ATRChannelStrategy(BaseStrategy):
    """
    ATR Channel Strategy
    
    This strategy uses ATR to create dynamic channels around a moving average,
    and generates signals when price moves outside these channels.
    """
    
    category = 'volatility'
    
    def __init__(self, ma_period=20, atr_period=14, atr_multiplier=2.0, logger=None):
        """
        Initialize the strategy
        
        Args:
            ma_period (int): Moving average period
            atr_period (int): ATR period
            atr_multiplier (float): ATR multiplier
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="ATRChannelStrategy",
            description=f"ATR Channel (MA{ma_period}, ATR{atr_period}×{atr_multiplier})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'ma_period': ma_period,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
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
            ma_period = self.parameters['ma_period']
            atr_period = self.parameters['atr_period']
            atr_multiplier = self.parameters['atr_multiplier']
            
            # Calculate moving average
            df = self.indicators.sma(df, [ma_period])
            df['MA'] = df[f'SMA_{ma_period}']
            
            # Calculate ATR
            df = self.indicators.atr(df, atr_period)
            
            # Calculate channels
            df['Upper_Channel'] = df['MA'] + df['ATR'] * atr_multiplier
            df['Lower_Channel'] = df['MA'] - df['ATR'] * atr_multiplier
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price crosses above lower channel from below
            df.loc[(df['Close'] > df['Lower_Channel']) & 
                   (df['Close'].shift(1) <= df['Lower_Channel'].shift(1)), 'Signal'] = 1
            
            # Sell signal: Price crosses below upper channel from above
            df.loc[(df['Close'] < df['Upper_Channel']) & 
                   (df['Close'].shift(1) >= df['Upper_Channel'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class VolatilityExpansionStrategy(BaseStrategy):
    """
    Volatility Expansion Strategy
    
    This strategy identifies periods of volatility expansion and generates signals
    based on the direction of the price movement during these periods.
    """
    
    category = 'volatility'
    
    def __init__(self, atr_period=14, atr_threshold=1.5, ma_period=10, logger=None):
        """
        Initialize the strategy
        
        Args:
            atr_period (int): ATR period
            atr_threshold (float): ATR threshold for volatility expansion
            ma_period (int): Moving average period for ATR
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="VolatilityExpansionStrategy",
            description=f"Volatility Expansion (ATR{atr_period}>{atr_threshold}×MA{ma_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'atr_period': atr_period,
            'atr_threshold': atr_threshold,
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
            atr_period = self.parameters['atr_period']
            atr_threshold = self.parameters['atr_threshold']
            ma_period = self.parameters['ma_period']
            
            # Calculate ATR
            df = self.indicators.atr(df, atr_period)
            
            # Calculate ATR moving average
            df['ATR_MA'] = df['ATR'].rolling(window=ma_period).mean()
            
            # Identify volatility expansion
            df['Volatility_Expansion'] = df['ATR'] > df['ATR_MA'] * atr_threshold
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Volatility expansion with price increase
            df.loc[(df['Volatility_Expansion']) & 
                   (df['Close'] > df['Close'].shift(1)), 'Signal'] = 1
            
            # Sell signal: Volatility expansion with price decrease
            df.loc[(df['Volatility_Expansion']) & 
                   (df['Close'] < df['Close'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner Channel Strategy
    
    This strategy uses Keltner Channels to identify potential reversal points,
    and generates signals when price moves outside the channels.
    """
    
    category = 'volatility'
    
    def __init__(self, ema_period=20, atr_period=10, atr_multiplier=2.0, logger=None):
        """
        Initialize the strategy
        
        Args:
            ema_period (int): EMA period
            atr_period (int): ATR period
            atr_multiplier (float): ATR multiplier
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="KeltnerChannelStrategy",
            description=f"Keltner Channel (EMA{ema_period}, ATR{atr_period}×{atr_multiplier})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'ema_period': ema_period,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
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
            ema_period = self.parameters['ema_period']
            atr_period = self.parameters['atr_period']
            atr_multiplier = self.parameters['atr_multiplier']
            
            # Calculate EMA
            df = self.indicators.ema(df, [ema_period])
            df['Middle_Line'] = df[f'EMA_{ema_period}']
            
            # Calculate ATR
            df = self.indicators.atr(df, atr_period)
            
            # Calculate Keltner Channels
            df['Upper_Channel'] = df['Middle_Line'] + df['ATR'] * atr_multiplier
            df['Lower_Channel'] = df['Middle_Line'] - df['ATR'] * atr_multiplier
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price crosses below lower channel and then back above it
            df.loc[(df['Close'] > df['Lower_Channel']) & 
                   (df['Close'].shift(1) <= df['Lower_Channel'].shift(1)) & 
                   (df['Close'].shift(2) < df['Lower_Channel'].shift(2)), 'Signal'] = 1
            
            # Sell signal: Price crosses above upper channel and then back below it
            df.loc[(df['Close'] < df['Upper_Channel']) & 
                   (df['Close'].shift(1) >= df['Upper_Channel'].shift(1)) & 
                   (df['Close'].shift(2) > df['Upper_Channel'].shift(2)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
