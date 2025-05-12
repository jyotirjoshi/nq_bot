#!/usr/bin/env python3
"""
Technical Indicators Module for NQ Alpha Elite

This module provides advanced technical indicators and market regime detection
for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import math

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.Technical")

class TechnicalIndicators:
    """
    Advanced technical indicators for NQ Alpha Elite
    
    This class provides a comprehensive set of technical indicators for
    analyzing market data and generating trading signals.
    """
    
    def __init__(self, logger=None):
        """
        Initialize technical indicators
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.Technical")
        
        # Load configuration
        self.config = config.INDICATORS_CONFIG
        
        # Initialize indicators
        self.indicators = {
            'rsi': self.rsi,
            'macd': self.macd,
            'bollinger_bands': self.bollinger_bands,
            'atr': self.atr,
            'ema': self.ema,
            'sma': self.sma,
            'stochastic': self.stochastic,
            'ichimoku': self.ichimoku,
            'vwap': self.vwap,
            'supertrend': self.supertrend,
            'adx': self.adx,
            'fibonacci': self.fibonacci,
            'pivot_points': self.pivot_points,
            'volume_profile': self.volume_profile,
            'market_profile': self.market_profile,
            'order_flow': self.order_flow
        }
        
        self.logger.info("Technical indicators initialized")
    
    def add_indicators(self, df, indicators=None):
        """
        Add technical indicators to DataFrame
        
        Args:
            df (DataFrame): Market data
            indicators (list, optional): List of indicators to add
            
        Returns:
            DataFrame: Market data with indicators
        """
        try:
            # Check if we have data
            if df is None or len(df) < 30:
                self.logger.warning("Insufficient data for indicators")
                return df
            
            # Make a copy to avoid modifying original
            df_copy = df.copy()
            
            # Ensure we have OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Map columns if needed
            for col in required_cols:
                if col not in df_copy.columns:
                    # Try to map from alternative column names
                    if col == 'Open' and 'price' in df_copy.columns:
                        df_copy['Open'] = df_copy['price'].shift(1)
                    elif col == 'High' and 'price' in df_copy.columns:
                        df_copy['High'] = df_copy['price']
                    elif col == 'Low' and 'price' in df_copy.columns:
                        df_copy['Low'] = df_copy['price']
                    elif col == 'Close' and 'price' in df_copy.columns:
                        df_copy['Close'] = df_copy['price']
                    elif col == 'Volume' and 'volume' in df_copy.columns:
                        df_copy['Volume'] = df_copy['volume']
            
            # Fill missing values
            for col in required_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill')
            
            # If no indicators specified, add all
            if indicators is None:
                indicators = ['rsi', 'macd', 'bollinger_bands', 'atr', 'ema', 'stochastic']
            
            # Add each indicator
            for indicator in indicators:
                if indicator in self.indicators:
                    try:
                        df_copy = self.indicators[indicator](df_copy)
                    except Exception as e:
                        self.logger.error(f"Error adding indicator {indicator}: {e}")
                else:
                    self.logger.warning(f"Unknown indicator: {indicator}")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            self.logger.error(traceback.format_exc())
            return df
    
    def rsi(self, df, period=None):
        """
        Relative Strength Index
        
        Args:
            df (DataFrame): Market data
            period (int, optional): RSI period
            
        Returns:
            DataFrame: Market data with RSI
        """
        try:
            # Use configured period if not provided
            period = period or self.config['rsi_period']
            
            # Check if we have Close prices
            if 'Close' not in df.columns:
                self.logger.warning("No Close prices for RSI calculation")
                return df
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            # Calculate RS
            rs = gain / loss
            
            # Calculate RSI
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return df
    
    def macd(self, df, fast_period=None, slow_period=None, signal_period=None):
        """
        Moving Average Convergence Divergence
        
        Args:
            df (DataFrame): Market data
            fast_period (int, optional): Fast EMA period
            slow_period (int, optional): Slow EMA period
            signal_period (int, optional): Signal EMA period
            
        Returns:
            DataFrame: Market data with MACD
        """
        try:
            # Use configured periods if not provided
            fast_period = fast_period or self.config['macd_fast']
            slow_period = slow_period or self.config['macd_slow']
            signal_period = signal_period or self.config['macd_signal']
            
            # Check if we have Close prices
            if 'Close' not in df.columns:
                self.logger.warning("No Close prices for MACD calculation")
                return df
            
            # Calculate EMAs
            ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            df['MACD'] = ema_fast - ema_slow
            
            # Calculate signal line
            df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return df
    
    def bollinger_bands(self, df, period=None, std_dev=None):
        """
        Bollinger Bands
        
        Args:
            df (DataFrame): Market data
            period (int, optional): Bollinger Bands period
            std_dev (float, optional): Standard deviation multiplier
            
        Returns:
            DataFrame: Market data with Bollinger Bands
        """
        try:
            # Use configured values if not provided
            period = period or self.config['bb_period']
            std_dev = std_dev or self.config['bb_std']
            
            # Check if we have Close prices
            if 'Close' not in df.columns:
                self.logger.warning("No Close prices for Bollinger Bands calculation")
                return df
            
            # Calculate middle band (SMA)
            df['BB_middle'] = df['Close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = df['Close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            df['BB_upper'] = df['BB_middle'] + (rolling_std * std_dev)
            df['BB_lower'] = df['BB_middle'] - (rolling_std * std_dev)
            
            # Calculate bandwidth
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # Calculate %B
            df['BB_pct_b'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def atr(self, df, period=None):
        """
        Average True Range
        
        Args:
            df (DataFrame): Market data
            period (int, optional): ATR period
            
        Returns:
            DataFrame: Market data with ATR
        """
        try:
            # Use configured period if not provided
            period = period or self.config['atr_period']
            
            # Check if we have OHLC prices
            required_cols = ['High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing OHLC data for ATR calculation")
                return df
            
            # Calculate true range
            df['tr0'] = abs(df['High'] - df['Low'])
            df['tr1'] = abs(df['High'] - df['Close'].shift())
            df['tr2'] = abs(df['Low'] - df['Close'].shift())
            df['TR'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            
            # Calculate ATR
            df['ATR'] = df['TR'].rolling(window=period).mean()
            
            # Clean up
            df = df.drop(['tr0', 'tr1', 'tr2', 'TR'], axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df
    
    def ema(self, df, periods=None):
        """
        Exponential Moving Average
        
        Args:
            df (DataFrame): Market data
            periods (list, optional): List of EMA periods
            
        Returns:
            DataFrame: Market data with EMAs
        """
        try:
            # Default periods if not provided
            periods = periods or [9, 21, 50, 200]
            
            # Check if we have Close prices
            if 'Close' not in df.columns:
                self.logger.warning("No Close prices for EMA calculation")
                return df
            
            # Calculate EMAs for each period
            for period in periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return df
    
    def sma(self, df, periods=None):
        """
        Simple Moving Average
        
        Args:
            df (DataFrame): Market data
            periods (list, optional): List of SMA periods
            
        Returns:
            DataFrame: Market data with SMAs
        """
        try:
            # Default periods if not provided
            periods = periods or [9, 21, 50, 200]
            
            # Check if we have Close prices
            if 'Close' not in df.columns:
                self.logger.warning("No Close prices for SMA calculation")
                return df
            
            # Calculate SMAs for each period
            for period in periods:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return df
