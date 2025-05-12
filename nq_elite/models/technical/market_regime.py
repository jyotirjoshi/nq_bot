#!/usr/bin/env python3
"""
Market Regime Detector for NQ Alpha Elite

This module provides advanced market regime detection capabilities for
trading NASDAQ 100 E-mini futures. It can identify different market regimes
such as trending, ranging, volatile, and quiet markets.
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
logger = logging.getLogger("NQAlpha.MarketRegime")

class MarketRegimeDetector:
    """
    Advanced market regime detector for NQ Alpha Elite
    
    This class provides methods to detect different market regimes
    and adapt trading strategies accordingly.
    """
    
    def __init__(self, lookback=100, logger=None):
        """
        Initialize market regime detector
        
        Args:
            lookback (int): Lookback period for regime detection
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.MarketRegime")
        self.lookback = lookback
        
        # Initialize regime state
        self.current_regime = None
        self.regime_history = []
        self.regime_start_time = None
        self.regime_duration = 0
        
        # Initialize regime thresholds
        self.thresholds = {
            'trend_strength': 25,  # ADX threshold for trend
            'volatility_high': 1.5,  # ATR ratio for high volatility
            'volatility_low': 0.5,  # ATR ratio for low volatility
            'range_width': 0.02,  # Price range width for ranging market (2%)
            'breakout_strength': 2.0,  # Bollinger Band width for breakout
            'chop_threshold': 60,  # Choppiness Index threshold
        }
        
        # Initialize regime characteristics
        self.characteristics = {
            'trending_up': {
                'description': 'Strong uptrend',
                'strategy': 'trend_following',
                'position_size': 1.0,
                'stop_loss': 2.0,
                'take_profit': 3.0
            },
            'trending_down': {
                'description': 'Strong downtrend',
                'strategy': 'trend_following',
                'position_size': 1.0,
                'stop_loss': 2.0,
                'take_profit': 3.0
            },
            'ranging': {
                'description': 'Range-bound market',
                'strategy': 'mean_reversion',
                'position_size': 0.7,
                'stop_loss': 1.5,
                'take_profit': 2.0
            },
            'volatile': {
                'description': 'High volatility market',
                'strategy': 'breakout',
                'position_size': 0.5,
                'stop_loss': 3.0,
                'take_profit': 4.0
            },
            'quiet': {
                'description': 'Low volatility market',
                'strategy': 'scalping',
                'position_size': 0.3,
                'stop_loss': 1.0,
                'take_profit': 1.5
            },
            'choppy': {
                'description': 'Choppy market',
                'strategy': 'neutral',
                'position_size': 0.2,
                'stop_loss': 1.0,
                'take_profit': 1.0
            },
            'breakout': {
                'description': 'Breakout market',
                'strategy': 'momentum',
                'position_size': 0.8,
                'stop_loss': 2.5,
                'take_profit': 3.5
            }
        }
        
        self.logger.info(f"Market Regime Detector initialized with {lookback} lookback period")
    
    def detect_regime(self, market_data):
        """
        Detect current market regime
        
        Args:
            market_data (DataFrame): Market data with indicators
            
        Returns:
            str: Current market regime
        """
        try:
            # Check if we have enough data
            if market_data is None or len(market_data) < self.lookback:
                self.logger.warning(f"Insufficient data for regime detection: {len(market_data) if market_data is not None else 0} < {self.lookback}")
                return None
            
            # Get recent data
            recent_data = market_data.iloc[-self.lookback:]
            
            # Calculate indicators if not present
            if 'ADX' not in recent_data.columns:
                recent_data = self._add_adx(recent_data)
            
            if 'ATR' not in recent_data.columns:
                recent_data = self._add_atr(recent_data)
            
            if 'BB_width' not in recent_data.columns:
                recent_data = self._add_bollinger_bands(recent_data)
            
            if 'CHOP' not in recent_data.columns:
                recent_data = self._add_choppiness_index(recent_data)
            
            # Get latest values
            latest = recent_data.iloc[-1]
            
            # Detect regime
            regime = self._classify_regime(recent_data, latest)
            
            # Update regime state
            if regime != self.current_regime:
                # Regime change
                self.regime_history.append((self.current_regime, self.regime_duration))
                self.current_regime = regime
                self.regime_start_time = datetime.now()
                self.regime_duration = 0
                
                self.logger.info(f"Market regime changed to: {regime}")
            else:
                # Same regime
                self.regime_duration += 1
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _classify_regime(self, data, latest):
        """
        Classify market regime based on indicators
        
        Args:
            data (DataFrame): Recent market data
            latest (Series): Latest data point
            
        Returns:
            str: Market regime
        """
        # Check for trending market
        if latest['ADX'] > self.thresholds['trend_strength']:
            # Determine trend direction
            if latest['DI_plus'] > latest['DI_minus']:
                return 'trending_up'
            else:
                return 'trending_down'
        
        # Check for volatile market
        atr_ratio = latest['ATR'] / data['ATR'].mean()
        if atr_ratio > self.thresholds['volatility_high']:
            # Check for breakout
            if latest['BB_width'] > self.thresholds['breakout_strength'] * data['BB_width'].mean():
                return 'breakout'
            else:
                return 'volatile'
        
        # Check for quiet market
        if atr_ratio < self.thresholds['volatility_low']:
            return 'quiet'
        
        # Check for choppy market
        if latest['CHOP'] > self.thresholds['chop_threshold']:
            return 'choppy'
        
        # Check for ranging market
        price_range = (data['High'].max() - data['Low'].min()) / latest['Close']
        if price_range < self.thresholds['range_width']:
            return 'ranging'
        
        # Default to choppy if no other regime detected
        return 'choppy'
    
    def get_regime_characteristics(self, regime=None):
        """
        Get characteristics of current or specified regime
        
        Args:
            regime (str, optional): Market regime
            
        Returns:
            dict: Regime characteristics
        """
        # Use current regime if not specified
        regime = regime or self.current_regime
        
        # Return characteristics if regime is valid
        if regime in self.characteristics:
            return self.characteristics[regime]
        else:
            return None
    
    def get_optimal_strategy(self, regime=None):
        """
        Get optimal trading strategy for current or specified regime
        
        Args:
            regime (str, optional): Market regime
            
        Returns:
            str: Optimal strategy
        """
        # Use current regime if not specified
        regime = regime or self.current_regime
        
        # Return strategy if regime is valid
        if regime in self.characteristics:
            return self.characteristics[regime]['strategy']
        else:
            return 'neutral'
    
    def get_position_sizing(self, regime=None):
        """
        Get optimal position sizing for current or specified regime
        
        Args:
            regime (str, optional): Market regime
            
        Returns:
            float: Position size multiplier (0.0-1.0)
        """
        # Use current regime if not specified
        regime = regime or self.current_regime
        
        # Return position size if regime is valid
        if regime in self.characteristics:
            return self.characteristics[regime]['position_size']
        else:
            return 0.5  # Default to 50% position size
    
    def get_risk_parameters(self, regime=None):
        """
        Get risk parameters for current or specified regime
        
        Args:
            regime (str, optional): Market regime
            
        Returns:
            dict: Risk parameters
        """
        # Use current regime if not specified
        regime = regime or self.current_regime
        
        # Return risk parameters if regime is valid
        if regime in self.characteristics:
            return {
                'stop_loss': self.characteristics[regime]['stop_loss'],
                'take_profit': self.characteristics[regime]['take_profit']
            }
        else:
            return {
                'stop_loss': 2.0,
                'take_profit': 2.0
            }
