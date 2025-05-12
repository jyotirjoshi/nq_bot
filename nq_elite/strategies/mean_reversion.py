#!/usr/bin/env python3
"""
Mean Reversion Strategies for NQ Alpha Elite

This module provides mean reversion strategies for trading NASDAQ 100 E-mini futures.
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

class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy
    
    This strategy generates buy signals when RSI is oversold and sell signals when RSI is overbought.
    """
    
    category = 'mean_reversion'
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, logger=None):
        """
        Initialize the strategy
        
        Args:
            rsi_period (int): RSI period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="RSIStrategy",
            description=f"RSI Strategy (RSI{rsi_period}, {oversold}/{overbought})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
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
            rsi_period = self.parameters['rsi_period']
            oversold = self.parameters['oversold']
            overbought = self.parameters['overbought']
            
            # Calculate RSI
            df = self.indicators.rsi(df, rsi_period)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: RSI crosses below oversold threshold
            df.loc[(df['RSI'] < oversold) & 
                   (df['RSI'].shift(1) >= oversold), 'Signal'] = 1
            
            # Sell signal: RSI crosses above overbought threshold
            df.loc[(df['RSI'] > overbought) & 
                   (df['RSI'].shift(1) <= overbought), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy
    
    This strategy generates buy signals when price touches the lower band and sell signals when price touches the upper band.
    """
    
    category = 'mean_reversion'
    
    def __init__(self, bb_period=20, bb_std=2, logger=None):
        """
        Initialize the strategy
        
        Args:
            bb_period (int): Bollinger Bands period
            bb_std (float): Bollinger Bands standard deviation
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="BollingerBandsStrategy",
            description=f"Bollinger Bands Strategy (BB{bb_period}, {bb_std}σ)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std
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
            bb_period = self.parameters['bb_period']
            bb_std = self.parameters['bb_std']
            
            # Calculate Bollinger Bands
            df = self.indicators.bollinger_bands(df, bb_period, bb_std)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Price touches or crosses below lower band
            df.loc[df['Close'] <= df['BB_lower'], 'Signal'] = 1
            
            # Sell signal: Price touches or crosses above upper band
            df.loc[df['Close'] >= df['BB_upper'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy
    
    This strategy generates buy signals when the stochastic oscillator is oversold and sell signals when it is overbought.
    """
    
    category = 'mean_reversion'
    
    def __init__(self, k_period=14, d_period=3, slowing=3, oversold=20, overbought=80, logger=None):
        """
        Initialize the strategy
        
        Args:
            k_period (int): %K period
            d_period (int): %D period
            slowing (int): Slowing period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="StochasticStrategy",
            description=f"Stochastic Strategy (%K{k_period}, %D{d_period}, {oversold}/{overbought})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'k_period': k_period,
            'd_period': d_period,
            'slowing': slowing,
            'oversold': oversold,
            'overbought': overbought
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
            k_period = self.parameters['k_period']
            d_period = self.parameters['d_period']
            slowing = self.parameters['slowing']
            oversold = self.parameters['oversold']
            overbought = self.parameters['overbought']
            
            # Calculate Stochastic
            df = self.indicators.stochastic(df, k_period, d_period, slowing)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: %K crosses above %D in oversold region
            df.loc[(df['Stoch_%K'] > df['Stoch_%D']) & 
                   (df['Stoch_%K'].shift(1) <= df['Stoch_%D'].shift(1)) & 
                   (df['Stoch_%K'] < oversold), 'Signal'] = 1
            
            # Sell signal: %K crosses below %D in overbought region
            df.loc[(df['Stoch_%K'] < df['Stoch_%D']) & 
                   (df['Stoch_%K'].shift(1) >= df['Stoch_%D'].shift(1)) & 
                   (df['Stoch_%K'] > overbought), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
