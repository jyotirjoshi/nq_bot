#!/usr/bin/env python3
"""
Paper Trading Module for NQ Alpha Elite

This module provides paper trading capabilities for the trading system,
allowing it to simulate trading strategies in real-time without real money.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import traceback
import math
import random

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

# Configure logging
logger = logging.getLogger("NQAlpha.PaperTrading")

class PaperTradingSimulator:
    """
    Paper Trading Simulator for NQ Alpha Elite
    
    This class provides paper trading capabilities for the trading system,
    allowing it to simulate trading strategies in real-time without real money.
    """
    
    def __init__(self, market_data_feed, initial_balance=100000, position_size=0.1, logger=None):
        """
        Initialize the paper trading simulator
        
        Args:
            market_data_feed: Market data feed
            initial_balance (float): Initial balance
            position_size (float): Position size as fraction of balance
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.PaperTrading")
        
        # Initialize market data feed
        self.market_data_feed = market_data_feed
        
        # Initialize account
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size = position_size
        
        # Initialize positions
        self.positions = []
        self.trades_history = []
        
        # Initialize state
        self.running = False
        self.thread = None
        self.last_update_time = None
        self.market_data = []
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        
        # Initialize metrics
        self.metrics = {
            'start_time': None,
            'trades': 0,
            'win_rate': 0.0,
            'return': 0.0,
            'drawdown': 0.0,
            'equity_curve': [initial_balance]
        }
        
        # Initialize signal generator
        self.signal_generator = None
        
        self.logger.info(f"Paper Trading Simulator initialized with ${initial_balance:.2f} balance")
    
    def start(self, update_interval=1.0):
        """
        Start the paper trading simulator
        
        Args:
            update_interval (float): Update interval in seconds
        """
        if self.running:
            self.logger.warning("Paper trading simulator already running")
            return
        
        self.logger.info(f"Starting paper trading simulator with {update_interval}s update interval")
        
        try:
            # Check if we have a market data feed
            if self.market_data_feed is None:
                self.logger.error("No market data feed available")
                return
            
            # Set running flag
            self.running = True
            self.metrics['start_time'] = datetime.now()
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._simulator_thread,
                args=(update_interval,),
                name="PaperTradingThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Paper trading thread started")
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting paper trading simulator: {e}")
    
    def stop(self):
        """Stop the paper trading simulator"""
        if not self.running:
            self.logger.warning("Paper trading simulator not running")
            return
        
        self.logger.info("Stopping paper trading simulator")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            
            # Close all positions
            self._close_all_positions()
            
            # Save results
            self._save_results()
            
            self.logger.info("Paper trading simulator stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping paper trading simulator: {e}")
    
    def _simulator_thread(self, update_interval):
        """
        Background thread for paper trading simulation
        
        Args:
            update_interval (float): Update interval in seconds
        """
        self.logger.info("Paper trading thread running")
        
        try:
            # Initialize variables
            last_update_time = time.time()
            last_print_time = time.time()
            
            while self.running:
                try:
                    # Get current time
                    current_time = time.time()
                    
                    # Check if it's time to update
                    if current_time - last_update_time >= update_interval:
                        # Update market data
                        self._update_market_data()
                        
                        # Update positions
                        self._update_positions()
                        
                        # Generate signals
                        self._generate_signals()
                        
                        # Execute trades
                        self._execute_trades()
                        
                        # Update last update time
                        last_update_time = current_time
                    
                    # Check if it's time to print metrics
                    if current_time - last_print_time >= 60:
                        # Print metrics
                        self._print_metrics()
                        
                        # Update last print time
                        last_print_time = current_time
                    
                    # Sleep for a short time
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in paper trading loop: {e}")
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"Fatal error in paper trading thread: {e}")
        
        self.logger.info("Paper trading thread stopped")
    
    def _update_market_data(self):
        """Update market data from feed"""
        try:
            # Check if we have a market data feed
            if self.market_data_feed is None:
                self.logger.warning("No market data feed available")
                return
            
            # Get latest data
            latest_data = self.market_data_feed.get_realtime_data()
            
            if latest_data is None:
                self.logger.debug("No new market data available")
                return
            
            # Add to market data
            self.market_data.append(latest_data)
            
            # Limit market data size
            max_data_points = 1000
            if len(self.market_data) > max_data_points:
                self.market_data = self.market_data[-max_data_points:]
            
            # Update last update time
            self.last_update_time = datetime.now()
            
            self.logger.debug(f"Updated market data: {len(self.market_data)} points")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _update_positions(self):
        """Update positions with current price"""
        try:
            # Check if we have market data
            if not self.market_data:
                return
            
            # Get latest data
            latest_data = self.market_data[-1]
            
            # Get current price
            if 'price' in latest_data:
                current_price = latest_data['price']
            else:
                self.logger.warning("No price data available")
                return
            
            # Update each position
            for position in self.positions:
                # Update current price
                position['current_price'] = current_price
                
                # Calculate P&L
                if position['side'] == 'long':
                    position['pnl'] = (current_price - position['entry_price']) * position['quantity']
                else:
                    position['pnl'] = (position['entry_price'] - current_price) * position['quantity']
            
            # Update equity curve
            total_pnl = sum(p['pnl'] for p in self.positions)
            current_equity = self.balance + total_pnl
            self.metrics['equity_curve'].append(current_equity)
            
            # Calculate return
            self.metrics['return'] = (current_equity / self.initial_balance - 1) * 100
            
            # Calculate drawdown
            peak = max(self.metrics['equity_curve'])
            current_drawdown = (peak - current_equity) / peak * 100
            self.metrics['drawdown'] = max(self.metrics['drawdown'], current_drawdown)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _generate_signals(self):
        """Generate trading signals"""
        try:
            # Check if we have market data
            if len(self.market_data) < 30:
                return
            
            # Convert market data to DataFrame
            df = pd.DataFrame(self.market_data)
            
            # Add technical indicators
            df = self.indicators.add_indicators(df)
            
            # Generate signals based on indicators
            if 'RSI' in df.columns and 'MACD' in df.columns:
                # RSI and MACD strategy
                df['Signal'] = 0  # Default to hold
                
                # Buy signal: RSI < 30 and MACD > MACD_signal
                df.loc[(df['RSI'] < 30) & (df['MACD'] > df['MACD_signal']), 'Signal'] = 1
                
                # Sell signal: RSI > 70 and MACD < MACD_signal
                df.loc[(df['RSI'] > 70) & (df['MACD'] < df['MACD_signal']), 'Signal'] = -1
            
            elif 'RSI' in df.columns:
                # RSI strategy
                df['Signal'] = 0  # Default to hold
                
                # Buy signal: RSI < 30
                df.loc[df['RSI'] < 30, 'Signal'] = 1
                
                # Sell signal: RSI > 70
                df.loc[df['RSI'] > 70, 'Signal'] = -1
            
            elif 'MACD' in df.columns:
                # MACD strategy
                df['Signal'] = 0  # Default to hold
                
                # Buy signal: MACD > MACD_signal
                df.loc[df['MACD'] > df['MACD_signal'], 'Signal'] = 1
                
                # Sell signal: MACD < MACD_signal
                df.loc[df['MACD'] < df['MACD_signal'], 'Signal'] = -1
            
            else:
                # No indicators available, use random signals for testing
                df['Signal'] = 0  # Default to hold
                
                # Random signals
                for i in range(len(df)):
                    if random.random() < 0.05:  # 5% chance of signal
                        df.loc[i, 'Signal'] = 1 if random.random() > 0.5 else -1
            
            # Store the signal generator
            self.signal_generator = df
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
    
    def _execute_trades(self):
        """Execute trades based on signals"""
        try:
            # Check if we have signals
            if self.signal_generator is None or len(self.signal_generator) < 1:
                return
            
            # Get latest signal
            latest_signal = self.signal_generator.iloc[-1]
            
            # Get current price
            if 'price' in latest_signal:
                current_price = latest_signal['price']
            else:
                self.logger.warning("No price data available")
                return
            
            # Get signal
            if 'Signal' in latest_signal:
                signal = latest_signal['Signal']
            else:
                self.logger.warning("No signal available")
                return
            
            # Execute trades based on signal
            if signal == 1:  # Buy signal
                # Check if we already have a long position
                if any(p['side'] == 'long' for p in self.positions):
                    self.logger.debug("Already have a long position, skipping buy")
                    return
                
                # Close any short positions
                self._close_positions('short')
                
                # Calculate position size
                position_capital = self.balance * self.position_size
                quantity = math.floor(position_capital / current_price)
                
                # Open long position
                if quantity > 0:
                    position = {
                        'side': 'long',
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'current_price': current_price,
                        'pnl': 0.0
                    }
                    
                    self.positions.append(position)
                    
                    self.logger.info(f"Opened long position: {quantity} @ {current_price:.2f}")
            
            elif signal == -1:  # Sell signal
                # Check if we already have a short position
                if any(p['side'] == 'short' for p in self.positions):
                    self.logger.debug("Already have a short position, skipping sell")
                    return
                
                # Close any long positions
                self._close_positions('long')
                
                # Calculate position size
                position_capital = self.balance * self.position_size
                quantity = math.floor(position_capital / current_price)
                
                # Open short position
                if quantity > 0:
                    position = {
                        'side': 'short',
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'current_price': current_price,
                        'pnl': 0.0
                    }
                    
                    self.positions.append(position)
                    
                    self.logger.info(f"Opened short position: {quantity} @ {current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
    
    def _close_positions(self, side=None):
        """
        Close positions
        
        Args:
            side (str, optional): Side to close ('long', 'short', or None for all)
        """
        try:
            # Get positions to close
            if side:
                positions_to_close = [p for p in self.positions if p['side'] == side]
            else:
                positions_to_close = self.positions.copy()
            
            # Close each position
            for position in positions_to_close:
                # Calculate P&L
                pnl = position['pnl']
                
                # Update balance
                self.balance += pnl
                
                # Add to trades history
                trade = {
                    'side': position['side'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': position['current_price'],
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'return': pnl / (position['entry_price'] * position['quantity'])
                }
                
                self.trades_history.append(trade)
                
                # Remove from positions
                self.positions.remove(position)
                
                self.logger.info(f"Closed {position['side']} position: {position['quantity']} @ {position['current_price']:.2f}, P&L: {pnl:.2f}")
            
            # Update metrics
            self.metrics['trades'] = len(self.trades_history)
            
            # Calculate win rate
            if self.trades_history:
                winning_trades = sum(1 for t in self.trades_history if t['pnl'] > 0)
                self.metrics['win_rate'] = winning_trades / len(self.trades_history)
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    def _close_all_positions(self):
        """Close all positions"""
        self._close_positions()
    
    def _print_metrics(self):
        """Print performance metrics"""
        try:
            # Calculate metrics
            runtime = datetime.now() - self.metrics['start_time'] if self.metrics['start_time'] else timedelta(0)
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            
            # Print metrics
            print("\n" + "=" * 50)
            print(f"  PAPER TRADING METRICS")
            print("=" * 50)
            print(f"Runtime: {runtime_str}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Return: {self.metrics['return']:.2f}%")
            print(f"Trades: {self.metrics['trades']}")
            print(f"Win Rate: {self.metrics['win_rate']:.2%}")
            print(f"Max Drawdown: {self.metrics['drawdown']:.2f}%")
            
            # Print active positions
            if self.positions:
                print("\nActive Positions:")
                for i, position in enumerate(self.positions):
                    print(f"  {i+1}. {position['side'].upper()}: {position['quantity']} @ {position['entry_price']:.2f}, P&L: {position['pnl']:.2f}")
            
            print("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Error printing metrics: {e}")
    
    def _save_results(self):
        """Save paper trading results"""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"paper_trading_{timestamp}.csv"
            filepath = os.path.join(config.DATA_DIR, "paper_trades", filename)
            
            # Create results DataFrame
            results = pd.DataFrame(self.trades_history)
            
            # Save to CSV
            results.to_csv(filepath, index=False)
            
            self.logger.info(f"Saved paper trading results to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def get_performance_summary(self):
        """
        Get performance summary
        
        Returns:
            dict: Performance summary
        """
        try:
            # Calculate metrics
            runtime = datetime.now() - self.metrics['start_time'] if self.metrics['start_time'] else timedelta(0)
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            
            # Create summary
            summary = {
                'runtime': runtime_str,
                'balance': self.balance,
                'return': self.metrics['return'],
                'trades': self.metrics['trades'],
                'win_rate': self.metrics['win_rate'],
                'drawdown': self.metrics['drawdown'],
                'active_positions': len(self.positions)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
