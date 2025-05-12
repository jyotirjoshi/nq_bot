#!/usr/bin/env python3
"""
Breakout Strategies for NQ Alpha Elite

This module provides breakout strategies for trading NASDAQ 100 E-mini futures.
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

class DonchianChannelBreakout(BaseStrategy):
    """
    Donchian Channel Breakout Strategy
    
    This strategy generates buy signals when price breaks above the upper Donchian channel,
    and sell signals when price breaks below the lower Donchian channel.
    """
    
    category = 'breakout'
    
    def __init__(self, period=20, logger=None):
        """
        Initialize the strategy
        
        Args:
            period (int): Donchian channel period
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="DonchianChannelBreakout",
            description=f"Donchian Channel Breakout ({period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'period': period
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
            period = self.parameters['period']
            
            # Calculate Donchian Channels
            df['Upper_Channel'] = df['High'].rolling(window=period).max()
            df['Lower_Channel'] = df['Low'].rolling(window=period).min()
            df['Middle_Channel'] = (df['Upper_Channel'] + df['Lower_Channel']) / 2
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price breaks above upper channel
            df.loc[(df['Close'] > df['Upper_Channel'].shift(1)), 'Signal'] = 1
            
            # Sell signal: Price breaks below lower channel
            df.loc[(df['Close'] < df['Lower_Channel'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout Strategy
    
    This strategy generates buy signals when price breaks above the previous day's high plus a volatility factor,
    and sell signals when price breaks below the previous day's low minus a volatility factor.
    """
    
    category = 'breakout'
    
    def __init__(self, atr_period=14, atr_multiplier=1.5, logger=None):
        """
        Initialize the strategy
        
        Args:
            atr_period (int): ATR period
            atr_multiplier (float): ATR multiplier
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="VolatilityBreakout",
            description=f"Volatility Breakout (ATR{atr_period}×{atr_multiplier})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
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
            atr_period = self.parameters['atr_period']
            atr_multiplier = self.parameters['atr_multiplier']
            
            # Calculate ATR
            df = self.indicators.atr(df, atr_period)
            
            # Calculate breakout levels
            df['Upper_Level'] = df['High'].shift(1) + df['ATR'] * atr_multiplier
            df['Lower_Level'] = df['Low'].shift(1) - df['ATR'] * atr_multiplier
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price breaks above upper level
            df.loc[df['Close'] > df['Upper_Level'], 'Signal'] = 1
            
            # Sell signal: Price breaks below lower level
            df.loc[df['Close'] < df['Lower_Level'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class PriceChannelBreakout(BaseStrategy):
    """
    Price Channel Breakout Strategy
    
    This strategy generates buy signals when price breaks above a resistance level,
    and sell signals when price breaks below a support level.
    """
    
    category = 'breakout'
    
    def __init__(self, period=20, confirmation_bars=2, logger=None):
        """
        Initialize the strategy
        
        Args:
            period (int): Price channel period
            confirmation_bars (int): Number of bars for confirmation
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="PriceChannelBreakout",
            description=f"Price Channel Breakout ({period}, {confirmation_bars} bars)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'period': period,
            'confirmation_bars': confirmation_bars
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
            period = self.parameters['period']
            confirmation_bars = self.parameters['confirmation_bars']
            
            # Calculate resistance and support levels
            df['Resistance'] = df['High'].rolling(window=period).max()
            df['Support'] = df['Low'].rolling(window=period).min()
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price breaks above resistance with confirmation
            for i in range(confirmation_bars, len(df)):
                if all(df['Close'].iloc[i-confirmation_bars:i+1] > df['Resistance'].iloc[i-confirmation_bars]):
                    df.loc[df.index[i], 'Signal'] = 1
            
            # Sell signal: Price breaks below support with confirmation
            for i in range(confirmation_bars, len(df)):
                if all(df['Close'].iloc[i-confirmation_bars:i+1] < df['Support'].iloc[i-confirmation_bars]):
                    df.loc[df.index[i], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
