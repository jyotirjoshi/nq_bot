#!/usr/bin/env python3
"""
Backtesting Module for NQ Alpha Elite

This module provides backtesting capabilities for the trading system,
allowing it to evaluate trading strategies on historical market data.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.Backtest")

def backtest_strategy(market_data, signal_column='Signal', initial_capital=100000, position_size=0.1):
    """
    Backtest a trading strategy on historical market data
    
    Args:
        market_data (DataFrame): Market data with signals
        signal_column (str): Column name for trading signals
        initial_capital (float): Initial capital
        position_size (float): Position size as fraction of capital
        
    Returns:
        dict: Backtest results
    """
    try:
        # Check if we have market data
        if market_data is None or len(market_data) < 2:
            logger.warning("Insufficient market data for backtesting")
            return None
        
        # Check if we have the signal column
        if signal_column not in market_data.columns:
            logger.warning(f"Signal column '{signal_column}' not found in market data")
            return None
        
        # Check if we have price data
        price_col = None
        for col in ['Close', 'price']:
            if col in market_data.columns:
                price_col = col
                break
        
        if price_col is None:
            logger.warning("No price column found in market data")
            return None
        
        # Create a copy of the market data
        df = market_data.copy()
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        # Run backtest
        for i in range(1, len(df)):
            # Get current and previous data
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Get current price
            current_price = current[price_col]
            
            # Get current signal
            signal = current[signal_column]
            
            # Update position value if we have a position
            if position != 0:
                # Calculate position value
                position_value = position * current_price
                
                # Calculate unrealized P&L
                if position > 0:  # Long position
                    unrealized_pnl = position * (current_price - entry_price)
                else:  # Short position
                    unrealized_pnl = position * (entry_price - current_price)
                
                # Update equity curve
                equity_curve.append(capital + unrealized_pnl)
            else:
                # No position, equity is just capital
                equity_curve.append(capital)
            
            # Execute trades based on signal
            if signal == 1 and position <= 0:  # Buy signal
                # Close any existing short position
                if position < 0:
                    # Calculate P&L
                    pnl = -position * (entry_price - current_price)
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': previous.name,
                        'exit_time': current.name,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'return': pnl / (abs(position) * entry_price)
                    }
                    
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                
                # Calculate position size
                position_capital = capital * position_size
                new_position = math.floor(position_capital / current_price)
                
                # Open long position
                if new_position > 0:
                    position = new_position
                    entry_price = current_price
            
            elif signal == -1 and position >= 0:  # Sell signal
                # Close any existing long position
                if position > 0:
                    # Calculate P&L
                    pnl = position * (current_price - entry_price)
                    
                    # Update capital
                    capital += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': previous.name,
                        'exit_time': current.name,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': pnl,
                        'return': pnl / (position * entry_price)
                    }
                    
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                
                # Calculate position size
                position_capital = capital * position_size
                new_position = -math.floor(position_capital / current_price)
                
                # Open short position
                if new_position < 0:
                    position = new_position
                    entry_price = current_price
        
        # Close any remaining position at the end
        if position != 0:
            # Get last price
            last_price = df.iloc[-1][price_col]
            
            # Calculate P&L
            if position > 0:  # Long position
                pnl = position * (last_price - entry_price)
            else:  # Short position
                pnl = -position * (entry_price - last_price)
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade = {
                'entry_time': df.iloc[-2].name,
                'exit_time': df.iloc[-1].name,
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': position,
                'pnl': pnl,
                'return': pnl / (abs(position) * entry_price)
            }
            
            trades.append(trade)
        
        # Calculate performance metrics
        total_return = (capital / initial_capital - 1) * 100
        
        # Calculate win rate
        if trades:
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0
        
        # Calculate drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Create results dictionary
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'trades': trades,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve
        }
        
        logger.info(f"Backtest completed: Return: {total_return:.2f}%, Trades: {len(trades)}, Win Rate: {win_rate:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        logger.error(traceback.format_exc())
        return None
