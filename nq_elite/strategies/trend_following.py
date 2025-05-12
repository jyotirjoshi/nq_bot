#!/usr/bin/env python3
"""
Trend Following Strategies for NQ Alpha Elite

This module provides trend following strategies for trading NASDAQ 100 E-mini futures.
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

class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when a fast moving average crosses above a slow moving average,
    and sell signals when the fast moving average crosses below the slow moving average.
    """
    
    category = 'trend_following'
    
    def __init__(self, fast_period=9, slow_period=21, signal_period=9, logger=None):
        """
        Initialize the strategy
        
        Args:
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
            signal_period (int): Signal smoothing period
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MovingAverageCrossover",
            description=f"Moving Average Crossover ({fast_period}/{slow_period}/{signal_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
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
            fast_period = self.parameters['fast_period']
            slow_period = self.parameters['slow_period']
            signal_period = self.parameters['signal_period']
            
            # Calculate EMAs
            df = self.indicators.ema(df, [fast_period, slow_period])
            
            # Calculate crossover
            df['Fast_EMA'] = df[f'EMA_{fast_period}']
            df['Slow_EMA'] = df[f'EMA_{slow_period}']
            df['EMA_Diff'] = df['Fast_EMA'] - df['Slow_EMA']
            
            # Smooth the difference
            df['EMA_Diff_Signal'] = df['EMA_Diff'].rolling(window=signal_period).mean()
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: EMA_Diff crosses above EMA_Diff_Signal
            df.loc[(df['EMA_Diff'] > df['EMA_Diff_Signal']) & 
                   (df['EMA_Diff'].shift(1) <= df['EMA_Diff_Signal'].shift(1)), 'Signal'] = 1
            
            # Sell signal: EMA_Diff crosses below EMA_Diff_Signal
            df.loc[(df['EMA_Diff'] < df['EMA_Diff_Signal']) & 
                   (df['EMA_Diff'].shift(1) >= df['EMA_Diff_Signal'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy
    
    This strategy generates buy signals when the MACD line crosses above the signal line,
    and sell signals when the MACD line crosses below the signal line.
    """
    
    category = 'trend_following'
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, logger=None):
        """
        Initialize the strategy
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MACDStrategy",
            description=f"MACD Strategy ({fast_period}/{slow_period}/{signal_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
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
            fast_period = self.parameters['fast_period']
            slow_period = self.parameters['slow_period']
            signal_period = self.parameters['signal_period']
            
            # Calculate MACD
            df = self.indicators.macd(df, fast_period, slow_period, signal_period)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: MACD crosses above signal line
            df.loc[(df['MACD'] > df['MACD_signal']) & 
                   (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'Signal'] = 1
            
            # Sell signal: MACD crosses below signal line
            df.loc[(df['MACD'] < df['MACD_signal']) & 
                   (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class ADXTrendStrategy(BaseStrategy):
    """
    ADX Trend Strategy
    
    This strategy uses the Average Directional Index (ADX) to identify strong trends,
    and generates signals based on the direction of the trend.
    """
    
    category = 'trend_following'
    
    def __init__(self, adx_period=14, adx_threshold=25, di_period=14, logger=None):
        """
        Initialize the strategy
        
        Args:
            adx_period (int): ADX period
            adx_threshold (int): ADX threshold for trend strength
            di_period (int): DI period
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="ADXTrendStrategy",
            description=f"ADX Trend Strategy (ADX{adx_period}>{adx_threshold}, DI{di_period})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'di_period': di_period
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
            adx_period = self.parameters['adx_period']
            adx_threshold = self.parameters['adx_threshold']
            di_period = self.parameters['di_period']
            
            # Calculate ADX
            df = self.indicators.adx(df, adx_period, di_period)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: ADX > threshold and +DI > -DI
            df.loc[(df['ADX'] > adx_threshold) & 
                   (df['DI_plus'] > df['DI_minus']), 'Signal'] = 1
            
            # Sell signal: ADX > threshold and +DI < -DI
            df.loc[(df['ADX'] > adx_threshold) & 
                   (df['DI_plus'] < df['DI_minus']), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
