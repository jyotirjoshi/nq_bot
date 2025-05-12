#!/usr/bin/env python3
"""
Multi-Timeframe Strategies for NQ Alpha Elite

This module provides multi-timeframe strategies for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

class MTFTrendStrategy(BaseStrategy):
    """
    Multi-Timeframe Trend Strategy
    
    This strategy combines trend signals from multiple timeframes to generate trading signals.
    """
    
    category = 'multi_timeframe'
    
    def __init__(self, fast_ma=9, slow_ma=21, higher_tf_factor=4, logger=None):
        """
        Initialize the strategy
        
        Args:
            fast_ma (int): Fast moving average period
            slow_ma (int): Slow moving average period
            higher_tf_factor (int): Higher timeframe factor
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MTFTrendStrategy",
            description=f"Multi-Timeframe Trend (MA{fast_ma}/{slow_ma}, Factor: {higher_tf_factor})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'higher_tf_factor': higher_tf_factor
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
            fast_ma = self.parameters['fast_ma']
            slow_ma = self.parameters['slow_ma']
            higher_tf_factor = self.parameters['higher_tf_factor']
            
            # Calculate moving averages for current timeframe
            df = self.indicators.ema(df, [fast_ma, slow_ma])
            df['Fast_MA'] = df[f'EMA_{fast_ma}']
            df['Slow_MA'] = df[f'EMA_{slow_ma}']
            
            # Calculate trend for current timeframe
            df['Current_Trend'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, -1)
            
            # Create higher timeframe data
            higher_tf_data = self._resample_to_higher_timeframe(df, higher_tf_factor)
            
            # Calculate moving averages for higher timeframe
            higher_tf_data = self.indicators.ema(higher_tf_data, [fast_ma, slow_ma])
            higher_tf_data['Fast_MA_H'] = higher_tf_data[f'EMA_{fast_ma}']
            higher_tf_data['Slow_MA_H'] = higher_tf_data[f'EMA_{slow_ma}']
            
            # Calculate trend for higher timeframe
            higher_tf_data['Higher_Trend'] = np.where(higher_tf_data['Fast_MA_H'] > higher_tf_data['Slow_MA_H'], 1, -1)
            
            # Merge higher timeframe trend back to original data
            df = self._merge_higher_timeframe_trend(df, higher_tf_data)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Both timeframes in uptrend
            df.loc[(df['Current_Trend'] == 1) & 
                   (df['Higher_Trend'] == 1) & 
                   (df['Current_Trend'].shift(1) <= 0), 'Signal'] = 1
            
            # Sell signal: Both timeframes in downtrend
            df.loc[(df['Current_Trend'] == -1) & 
                   (df['Higher_Trend'] == -1) & 
                   (df['Current_Trend'].shift(1) >= 0), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _resample_to_higher_timeframe(self, df, factor):
        """
        Resample data to higher timeframe
        
        Args:
            df (DataFrame): Market data
            factor (int): Resampling factor
            
        Returns:
            DataFrame: Resampled data
        """
        try:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Create a synthetic datetime index
                start_date = datetime.now() - timedelta(days=len(df))
                df = df.copy()
                df.index = pd.date_range(start=start_date, periods=len(df), freq='T')
            
            # Determine resampling rule
            if 'T' in df.index.freq.name:
                # Minutes
                resample_rule = f'{factor}T'
            elif 'H' in df.index.freq.name:
                # Hours
                resample_rule = f'{factor}H'
            elif 'D' in df.index.freq.name:
                # Days
                resample_rule = f'{factor}D'
            else:
                # Default to minutes
                resample_rule = f'{factor}T'
            
            # Resample data
            resampled = df.resample(resample_rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return df
    
    def _merge_higher_timeframe_trend(self, df, higher_tf_data):
        """
        Merge higher timeframe trend back to original data
        
        Args:
            df (DataFrame): Original market data
            higher_tf_data (DataFrame): Higher timeframe data
            
        Returns:
            DataFrame: Merged data
        """
        try:
            # Forward fill higher timeframe trend
            higher_tf_trend = higher_tf_data['Higher_Trend'].reindex(
                df.index, method='ffill'
            )
            
            # Add to original data
            df['Higher_Trend'] = higher_tf_trend
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error merging higher timeframe trend: {e}")
            return df


class MTFMomentumStrategy(BaseStrategy):
    """
    Multi-Timeframe Momentum Strategy
    
    This strategy combines momentum indicators from multiple timeframes to generate trading signals.
    """
    
    category = 'multi_timeframe'
    
    def __init__(self, rsi_period=14, higher_tf_factor=4, logger=None):
        """
        Initialize the strategy
        
        Args:
            rsi_period (int): RSI period
            higher_tf_factor (int): Higher timeframe factor
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MTFMomentumStrategy",
            description=f"Multi-Timeframe Momentum (RSI{rsi_period}, Factor: {higher_tf_factor})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'rsi_period': rsi_period,
            'higher_tf_factor': higher_tf_factor
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
            higher_tf_factor = self.parameters['higher_tf_factor']
            
            # Calculate RSI for current timeframe
            df = self.indicators.rsi(df, rsi_period)
            
            # Create higher timeframe data
            higher_tf_data = self._resample_to_higher_timeframe(df, higher_tf_factor)
            
            # Calculate RSI for higher timeframe
            higher_tf_data = self.indicators.rsi(higher_tf_data, rsi_period)
            
            # Merge higher timeframe RSI back to original data
            df = self._merge_higher_timeframe_rsi(df, higher_tf_data)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Current RSI oversold and higher timeframe RSI rising
            df.loc[(df['RSI'] < 30) & 
                   (df['Higher_RSI'] > df['Higher_RSI'].shift(1)), 'Signal'] = 1
            
            # Sell signal: Current RSI overbought and higher timeframe RSI falling
            df.loc[(df['RSI'] > 70) & 
                   (df['Higher_RSI'] < df['Higher_RSI'].shift(1)), 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _resample_to_higher_timeframe(self, df, factor):
        """
        Resample data to higher timeframe
        
        Args:
            df (DataFrame): Market data
            factor (int): Resampling factor
            
        Returns:
            DataFrame: Resampled data
        """
        try:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Create a synthetic datetime index
                start_date = datetime.now() - timedelta(days=len(df))
                df = df.copy()
                df.index = pd.date_range(start=start_date, periods=len(df), freq='T')
            
            # Determine resampling rule
            if 'T' in df.index.freq.name:
                # Minutes
                resample_rule = f'{factor}T'
            elif 'H' in df.index.freq.name:
                # Hours
                resample_rule = f'{factor}H'
            elif 'D' in df.index.freq.name:
                # Days
                resample_rule = f'{factor}D'
            else:
                # Default to minutes
                resample_rule = f'{factor}T'
            
            # Resample data
            resampled = df.resample(resample_rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return df
    
    def _merge_higher_timeframe_rsi(self, df, higher_tf_data):
        """
        Merge higher timeframe RSI back to original data
        
        Args:
            df (DataFrame): Original market data
            higher_tf_data (DataFrame): Higher timeframe data
            
        Returns:
            DataFrame: Merged data
        """
        try:
            # Forward fill higher timeframe RSI
            higher_tf_rsi = higher_tf_data['RSI'].reindex(
                df.index, method='ffill'
            )
            
            # Add to original data
            df['Higher_RSI'] = higher_tf_rsi
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error merging higher timeframe RSI: {e}")
            return df
