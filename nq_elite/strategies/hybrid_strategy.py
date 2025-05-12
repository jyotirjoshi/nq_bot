#!/usr/bin/env python3
"""
Hybrid Strategies for NQ Alpha Elite

This module provides hybrid strategies that combine multiple approaches for trading NASDAQ 100 E-mini futures.
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
from nq_alpha_elite.models.rl.base_agent import NQRLAgent

class TechnicalRLHybridStrategy(BaseStrategy):
    """
    Technical Analysis and Reinforcement Learning Hybrid Strategy
    
    This strategy combines traditional technical analysis with reinforcement learning
    to generate trading signals.
    """
    
    category = 'hybrid'
    
    def __init__(self, rl_agent=None, technical_weight=0.5, rl_weight=0.5, logger=None):
        """
        Initialize the strategy
        
        Args:
            rl_agent (NQRLAgent, optional): Reinforcement learning agent
            technical_weight (float): Weight for technical signals
            rl_weight (float): Weight for RL signals
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="TechnicalRLHybridStrategy",
            description=f"Technical-RL Hybrid (Tech: {technical_weight}, RL: {rl_weight})",
            logger=logger
        )
        
        # Set parameters
        self.parameters = {
            'technical_weight': technical_weight,
            'rl_weight': rl_weight
        }
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
        
        # Initialize RL agent
        self.rl_agent = rl_agent or NQRLAgent(logger=self.logger)
    
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
            technical_weight = self.parameters['technical_weight']
            rl_weight = self.parameters['rl_weight']
            
            # Generate technical signals
            df = self._generate_technical_signals(df)
            
            # Generate RL signals
            df = self._generate_rl_signals(df)
            
            # Combine signals
            df['Combined_Signal'] = (
                df['Technical_Signal'] * technical_weight + 
                df['RL_Signal'] * rl_weight
            )
            
            # Threshold combined signal
            df['Signal'] = 0  # Default to hold
            df.loc[df['Combined_Signal'] > 0.3, 'Signal'] = 1  # Buy
            df.loc[df['Combined_Signal'] < -0.3, 'Signal'] = -1  # Sell
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _generate_technical_signals(self, df):
        """
        Generate technical analysis signals
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with technical signals
        """
        try:
            # Add technical indicators
            df = self.indicators.rsi(df)
            df = self.indicators.macd(df)
            df = self.indicators.bollinger_bands(df)
            df = self.indicators.sma(df, [9, 21, 50])
            
            # Initialize technical signal
            df['Technical_Signal'] = 0  # Default to hold
            
            # RSI signals
            df.loc[df['RSI'] < 30, 'Technical_Signal'] += 0.3  # Oversold
            df.loc[df['RSI'] > 70, 'Technical_Signal'] -= 0.3  # Overbought
            
            # MACD signals
            df.loc[df['MACD'] > df['MACD_signal'], 'Technical_Signal'] += 0.3  # Bullish
            df.loc[df['MACD'] < df['MACD_signal'], 'Technical_Signal'] -= 0.3  # Bearish
            
            # Bollinger Bands signals
            df.loc[df['Close'] < df['BB_lower'], 'Technical_Signal'] += 0.2  # Oversold
            df.loc[df['Close'] > df['BB_upper'], 'Technical_Signal'] -= 0.2  # Overbought
            
            # Moving average signals
            df.loc[df['SMA_9'] > df['SMA_21'], 'Technical_Signal'] += 0.2  # Bullish
            df.loc[df['SMA_9'] < df['SMA_21'], 'Technical_Signal'] -= 0.2  # Bearish
            
            # Clip signal to [-1, 1]
            df['Technical_Signal'] = df['Technical_Signal'].clip(-1, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating technical signals: {e}")
            return df
    
    def _generate_rl_signals(self, df):
        """
        Generate reinforcement learning signals
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with RL signals
        """
        try:
            # Initialize RL signal
            df['RL_Signal'] = 0  # Default to hold
            
            # Extract features for RL state
            for i in range(len(df)):
                if i < 30:  # Skip first few bars
                    continue
                
                # Get state from recent data
                state = self._extract_features(df.iloc[i-30:i])
                
                # Get action from RL agent
                action = self.rl_agent.act(state)
                
                # Convert action to signal
                if action == 0:  # Buy
                    df.loc[df.index[i], 'RL_Signal'] = 1
                elif action == 1:  # Sell
                    df.loc[df.index[i], 'RL_Signal'] = -1
                # else: Hold (0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating RL signals: {e}")
            return df
    
    def _extract_features(self, recent_data):
        """
        Extract features for RL state
        
        Args:
            recent_data (DataFrame): Recent market data
            
        Returns:
            numpy.ndarray: State representation
        """
        try:
            # Extract basic features
            features = []
            
            # Price changes
            price_changes = recent_data['Close'].pct_change().dropna().values
            features.extend(price_changes[-5:])  # Last 5 price changes
            
            # Technical indicators
            if 'RSI' in recent_data.columns:
                features.append(recent_data['RSI'].iloc[-1] / 100)  # Normalize RSI
            
            if 'MACD' in recent_data.columns and 'MACD_signal' in recent_data.columns:
                features.append(recent_data['MACD'].iloc[-1] - recent_data['MACD_signal'].iloc[-1])
            
            # Moving averages
            for ma in ['SMA_9', 'SMA_21', 'SMA_50']:
                if ma in recent_data.columns:
                    features.append(recent_data['Close'].iloc[-1] / recent_data[ma].iloc[-1] - 1)
            
            # Pad or truncate to fixed length
            state_size = self.rl_agent.state_size
            if len(features) < state_size:
                features.extend([0] * (state_size - len(features)))
            elif len(features) > state_size:
                features = features[:state_size]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(self.rl_agent.state_size)


class RegimeSwitchingStrategy(BaseStrategy):
    """
    Regime Switching Strategy
    
    This strategy switches between different sub-strategies based on the detected market regime.
    """
    
    category = 'hybrid'
    
    def __init__(self, logger=None):
        """
        Initialize the strategy
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        super().__init__(
            name="RegimeSwitchingStrategy",
            description="Regime Switching Strategy",
            logger=logger
        )
        
        # Initialize indicators
        self.indicators = TechnicalIndicators(logger=self.logger)
        
        # Initialize sub-strategies
        self.sub_strategies = {
            'trending_up': self._trend_following_strategy,
            'trending_down': self._trend_following_strategy,
            'ranging': self._mean_reversion_strategy,
            'volatile': self._breakout_strategy,
            'quiet': self._scalping_strategy,
            'choppy': self._neutral_strategy,
            'breakout': self._momentum_strategy
        }
    
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
            
            # Add technical indicators
            df = self.indicators.add_indicators(df)
            
            # Detect market regime
            df = self._detect_market_regime(df)
            
            # Generate signals based on regime
            for i in range(len(df)):
                if i < 50:  # Skip first few bars
                    continue
                
                # Get current regime
                regime = df['Market_Regime'].iloc[i]
                
                # Get strategy for regime
                strategy_func = self.sub_strategies.get(regime, self._neutral_strategy)
                
                # Generate signal
                signal = strategy_func(df.iloc[:i+1])
                
                # Set signal
                df.loc[df.index[i], 'Signal'] = signal
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return market_data
    
    def _detect_market_regime(self, df):
        """
        Detect market regime
        
        Args:
            df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with regime
        """
        try:
            # Calculate volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Calculate trend strength
            df['Trend_Strength'] = abs(df['SMA_50'] / df['SMA_200'] - 1) * 100
            
            # Calculate range
            df['Range_Width'] = (df['BB_upper'] - df['BB_lower']) / df['Close'] * 100
            
            # Detect regime
            df['Market_Regime'] = 'neutral'  # Default
            
            # Trending up
            df.loc[(df['SMA_50'] > df['SMA_200']) & 
                   (df['Trend_Strength'] > 5), 'Market_Regime'] = 'trending_up'
            
            # Trending down
            df.loc[(df['SMA_50'] < df['SMA_200']) & 
                   (df['Trend_Strength'] > 5), 'Market_Regime'] = 'trending_down'
            
            # Ranging
            df.loc[(df['Trend_Strength'] < 2) & 
                   (df['Volatility'] < 0.2), 'Market_Regime'] = 'ranging'
            
            # Volatile
            df.loc[df['Volatility'] > 0.3, 'Market_Regime'] = 'volatile'
            
            # Quiet
            df.loc[df['Volatility'] < 0.1, 'Market_Regime'] = 'quiet'
            
            # Choppy
            df.loc[(df['Trend_Strength'] < 1) & 
                   (df['Volatility'] > 0.15), 'Market_Regime'] = 'choppy'
            
            # Breakout
            df.loc[(df['Close'] > df['BB_upper']) | 
                   (df['Close'] < df['BB_lower']), 'Market_Regime'] = 'breakout'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return df
    
    def _trend_following_strategy(self, df):
        """Trend following strategy for trending regimes"""
        try:
            # Use MACD for trend following
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
                    return 1  # Buy
                elif df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]:
                    return -1  # Sell
            
            return 0  # Hold
        except:
            return 0
    
    def _mean_reversion_strategy(self, df):
        """Mean reversion strategy for ranging regimes"""
        try:
            # Use RSI for mean reversion
            if 'RSI' in df.columns:
                if df['RSI'].iloc[-1] < 30:
                    return 1  # Buy
                elif df['RSI'].iloc[-1] > 70:
                    return -1  # Sell
            
            return 0  # Hold
        except:
            return 0
    
    def _breakout_strategy(self, df):
        """Breakout strategy for volatile regimes"""
        try:
            # Use Bollinger Bands for breakout
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
                    return 1  # Buy
                elif df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
                    return -1  # Sell
            
            return 0  # Hold
        except:
            return 0
    
    def _scalping_strategy(self, df):
        """Scalping strategy for quiet regimes"""
        try:
            # Use short-term price action for scalping
            if len(df) >= 3:
                if (df['Close'].iloc[-1] > df['Close'].iloc[-2] and 
                    df['Close'].iloc[-2] > df['Close'].iloc[-3]):
                    return 1  # Buy
                elif (df['Close'].iloc[-1] < df['Close'].iloc[-2] and 
                      df['Close'].iloc[-2] < df['Close'].iloc[-3]):
                    return -1  # Sell
            
            return 0  # Hold
        except:
            return 0
    
    def _neutral_strategy(self, df):
        """Neutral strategy for choppy regimes"""
        return 0  # Hold
    
    def _momentum_strategy(self, df):
        """Momentum strategy for breakout regimes"""
        try:
            # Use ADX for momentum
            if 'ADX' in df.columns:
                if df['ADX'].iloc[-1] > 25:
                    if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                        return 1  # Buy
                    else:
                        return -1  # Sell
            
            return 0  # Hold
        except:
            return 0
