#!/usr/bin/env python3
"""
Base Strategy Class for NQ Alpha Elite

This module provides the base class for all trading strategies.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    This abstract class defines the interface for all trading strategies.
    All strategies must inherit from this class and implement the required methods.
    """
    
    def __init__(self, name=None, description=None, logger=None):
        """
        Initialize the strategy
        
        Args:
            name (str, optional): Strategy name
            description (str, optional): Strategy description
            logger (logging.Logger, optional): Logger instance
        """
        self.name = name or self.__class__.__name__
        self.description = description or f"{self.name} Strategy"
        self.logger = logger or logging.getLogger(f"NQAlpha.Strategy.{self.name}")
        
        # Initialize performance metrics
        self.metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_trade': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0
        }
        
        # Initialize parameters
        self.parameters = {}
        
        # Initialize trades
        self.trades = []
        
        self.logger.info(f"Strategy {self.name} initialized")
    
    @abstractmethod
    def generate_signals(self, market_data):
        """
        Generate trading signals
        
        Args:
            market_data (DataFrame): Market data with indicators
            
        Returns:
            DataFrame: Market data with signals
        """
        pass
    
    def preprocess_data(self, market_data):
        """
        Preprocess market data
        
        Args:
            market_data (DataFrame): Raw market data
            
        Returns:
            DataFrame: Preprocessed market data
        """
        return market_data
    
    def postprocess_signals(self, market_data):
        """
        Postprocess signals
        
        Args:
            market_data (DataFrame): Market data with signals
            
        Returns:
            DataFrame: Market data with processed signals
        """
        return market_data
    
    def calculate_metrics(self, backtest_results):
        """
        Calculate performance metrics
        
        Args:
            backtest_results (dict): Backtest results
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            # Calculate metrics
            if trades:
                # Total trades
                total_trades = len(trades)
                
                # Profitable trades
                profitable_trades = sum(1 for trade in trades if trade['pnl'] > 0)
                
                # Losing trades
                losing_trades = sum(1 for trade in trades if trade['pnl'] <= 0)
                
                # Win rate
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
                
                # Profit factor
                gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
                gross_loss = sum(abs(trade['pnl']) for trade in trades if trade['pnl'] < 0)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Average trade
                avg_trade = sum(trade['pnl'] for trade in trades) / total_trades if total_trades > 0 else 0.0
                
                # Update metrics
                self.metrics.update({
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_trade': avg_trade,
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'losing_trades': losing_trades
                })
                
                # Update trades
                self.trades = trades
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return self.metrics
    
    def optimize_parameters(self, market_data, parameter_grid, metric='profit_factor'):
        """
        Optimize strategy parameters
        
        Args:
            market_data (DataFrame): Market data
            parameter_grid (dict): Parameter grid for optimization
            metric (str): Metric to optimize
            
        Returns:
            dict: Optimal parameters
        """
        try:
            from itertools import product
            
            # Initialize best parameters and metric
            best_parameters = None
            best_metric_value = float('-inf')
            
            # Generate parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            param_combinations = list(product(*param_values))
            
            # Iterate over parameter combinations
            for params in param_combinations:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Update parameters
                self.parameters.update(param_dict)
                
                # Generate signals
                signals = self.generate_signals(market_data.copy())
                
                # Run backtest
                from nq_alpha_elite.execution.backtest import backtest_strategy
                backtest_results = backtest_strategy(signals, 'Signal')
                
                # Calculate metrics
                metrics = self.calculate_metrics(backtest_results)
                
                # Check if better
                if metrics[metric] > best_metric_value:
                    best_metric_value = metrics[metric]
                    best_parameters = param_dict.copy()
            
            # Update parameters
            if best_parameters:
                self.parameters.update(best_parameters)
                self.logger.info(f"Optimized parameters: {best_parameters}, {metric}: {best_metric_value}")
            
            return best_parameters
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return self.parameters
