#!/usr/bin/env python3
"""
Specialized Strategies for NQ Alpha Elite

This module provides specialized strategies for trading NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

class GapFillStrategy(BaseStrategy):
    """
    Gap Fill Strategy
    
    This strategy identifies price gaps and generates signals to trade the gap fill.
    """
    
    category = 'specialized'
    
    def __init__(self, min_gap_pct=0.5, max_gap_pct=3.0, logger=None):
        """
        Initialize the strategy
        
        Args:
            min_gap_pct (float): Minimum gap size as percentage
            max_gap_pct (float): Maximum gap size as percentage
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="GapFillStrategy",
            description=f"Gap Fill Strategy (Gap: {min_gap_pct}%-{max_gap_pct}%)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'min_gap_pct': min_gap_pct,
            'max_gap_pct': max_gap_pct
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
            min_gap_pct = self.parameters['min_gap_pct']
            max_gap_pct = self.parameters['max_gap_pct']
            
            # Calculate gaps
            df['Previous_Close'] = df['Close'].shift(1)
            df['Gap'] = (df['Open'] - df['Previous_Close']) / df['Previous_Close'] * 100
            
            # Identify gap up and gap down
            df['Gap_Up'] = (df['Gap'] > min_gap_pct) & (df['Gap'] < max_gap_pct)
            df['Gap_Down'] = (df['Gap'] < -min_gap_pct) & (df['Gap'] > -max_gap_pct)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Gap down
            df.loc[df['Gap_Down'], 'Signal'] = 1
            
            # Sell signal: Gap up
            df.loc[df['Gap_Up'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Breakout Strategy
    
    This strategy identifies breakouts with high volume and generates signals accordingly.
    """
    
    category = 'specialized'
    
    def __init__(self, volume_threshold=2.0, price_threshold=1.0, logger=None):
        """
        Initialize the strategy
        
        Args:
            volume_threshold (float): Volume threshold as multiple of average
            price_threshold (float): Price threshold as percentage
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="VolumeBreakoutStrategy",
            description=f"Volume Breakout Strategy (Vol: {volume_threshold}×, Price: {price_threshold}%)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'volume_threshold': volume_threshold,
            'price_threshold': price_threshold
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
            volume_threshold = self.parameters['volume_threshold']
            price_threshold = self.parameters['price_threshold']
            
            # Calculate volume and price metrics
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['Price_Change'] = df['Close'].pct_change() * 100
            
            # Identify high volume breakouts
            df['High_Volume_Up'] = (
                (df['Volume_Ratio'] > volume_threshold) & 
                (df['Price_Change'] > price_threshold)
            )
            
            df['High_Volume_Down'] = (
                (df['Volume_Ratio'] > volume_threshold) & 
                (df['Price_Change'] < -price_threshold)
            )
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: High volume up breakout
            df.loc[df['High_Volume_Up'], 'Signal'] = 1
            
            # Sell signal: High volume down breakout
            df.loc[df['High_Volume_Down'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class SwingHighLowStrategy(BaseStrategy):
    """
    Swing High Low Strategy
    
    This strategy identifies swing highs and lows and generates signals based on breakouts.
    """
    
    category = 'specialized'
    
    def __init__(self, lookback=5, confirmation=2, logger=None):
        """
        Initialize the strategy
        
        Args:
            lookback (int): Number of bars to look back for swing points
            confirmation (int): Number of bars for confirmation
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="SwingHighLowStrategy",
            description=f"Swing High Low Strategy (Lookback: {lookback}, Confirmation: {confirmation})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'lookback': lookback,
            'confirmation': confirmation
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
            confirmation = self.parameters['confirmation']
            
            # Identify swing highs and lows
            df['Swing_High'] = False
            df['Swing_Low'] = False
            
            for i in range(lookback, len(df) - lookback):
                # Check for swing high
                if all(df['High'].iloc[i] > df['High'].iloc[i-lookback:i]) and \
                   all(df['High'].iloc[i] > df['High'].iloc[i+1:i+lookback+1]):
                    df.loc[df.index[i], 'Swing_High'] = True
                
                # Check for swing low
                if all(df['Low'].iloc[i] < df['Low'].iloc[i-lookback:i]) and \
                   all(df['Low'].iloc[i] < df['Low'].iloc[i+1:i+lookback+1]):
                    df.loc[df.index[i], 'Swing_Low'] = True
            
            # Find most recent swing points
            df['Last_Swing_High'] = np.nan
            df['Last_Swing_Low'] = np.nan
            
            last_high = np.nan
            last_low = np.nan
            
            for i in range(len(df)):
                if df['Swing_High'].iloc[i]:
                    last_high = df['High'].iloc[i]
                
                if df['Swing_Low'].iloc[i]:
                    last_low = df['Low'].iloc[i]
                
                df.loc[df.index[i], 'Last_Swing_High'] = last_high
                df.loc[df.index[i], 'Last_Swing_Low'] = last_low
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Break above swing high with confirmation
            for i in range(confirmation, len(df)):
                if all(df['Close'].iloc[i-confirmation:i+1] > df['Last_Swing_High'].iloc[i-confirmation]):
                    df.loc[df.index[i], 'Signal'] = 1
            
            # Sell signal: Break below swing low with confirmation
            for i in range(confirmation, len(df)):
                if all(df['Close'].iloc[i-confirmation:i+1] < df['Last_Swing_Low'].iloc[i-confirmation]):
                    df.loc[df.index[i], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data


class MarketOpenStrategy(BaseStrategy):
    """
    Market Open Strategy
    
    This strategy trades the first hour of the market session based on the opening range.
    """
    
    category = 'specialized'
    
    def __init__(self, range_minutes=30, breakout_pct=0.2, logger=None):
        """
        Initialize the strategy
        
        Args:
            range_minutes (int): Minutes to establish opening range
            breakout_pct (float): Breakout threshold as percentage
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="MarketOpenStrategy",
            description=f"Market Open Strategy (Range: {range_minutes}min, Breakout: {breakout_pct}%)",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'range_minutes': range_minutes,
            'breakout_pct': breakout_pct
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
            range_minutes = self.parameters['range_minutes']
            breakout_pct = self.parameters['breakout_pct']
            
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Cannot determine market open without datetime index
                return df
            
            # Define market open time (9:30 AM ET)
            market_open = time(9, 30)
            
            # Initialize columns
            df['Market_Open'] = False
            df['Opening_Range_High'] = np.nan
            df['Opening_Range_Low'] = np.nan
            
            # Process each day
            for day in df.index.date.unique():
                # Get data for this day
                day_data = df[df.index.date == day]
                
                # Find market open
                market_open_idx = None
                for i, idx in enumerate(day_data.index):
                    if idx.time() >= market_open:
                        market_open_idx = i
                        break
                
                if market_open_idx is None:
                    continue
                
                # Mark market open
                df.loc[day_data.index[market_open_idx], 'Market_Open'] = True
                
                # Calculate opening range
                range_end_idx = min(market_open_idx + range_minutes, len(day_data) - 1)
                opening_range = day_data.iloc[market_open_idx:range_end_idx+1]
                
                if len(opening_range) == 0:
                    continue
                
                opening_range_high = opening_range['High'].max()
                opening_range_low = opening_range['Low'].min()
                
                # Set opening range for the day
                df.loc[day_data.index, 'Opening_Range_High'] = opening_range_high
                df.loc[day_data.index, 'Opening_Range_Low'] = opening_range_low
            
            # Forward fill opening range
            df['Opening_Range_High'] = df['Opening_Range_High'].fillna(method='ffill')
            df['Opening_Range_Low'] = df['Opening_Range_Low'].fillna(method='ffill')
            
            # Calculate breakout thresholds
            df['Upper_Breakout'] = df['Opening_Range_High'] * (1 + breakout_pct / 100)
            df['Lower_Breakout'] = df['Opening_Range_Low'] * (1 - breakout_pct / 100)
            
            # Generate signals
            df['Signal'] = 0  # Default to hold
            
            # Buy signal: Break above opening range high
            df.loc[df['Close'] > df['Upper_Breakout'], 'Signal'] = 1
            
            # Sell signal: Break below opening range low
            df.loc[df['Close'] < df['Lower_Breakout'], 'Signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
