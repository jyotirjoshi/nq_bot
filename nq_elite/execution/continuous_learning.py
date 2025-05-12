#!/usr/bin/env python3
"""
Continuous Learning Module for NQ Alpha Elite

This module provides continuous learning capabilities for the trading system,
allowing it to learn from live market data in real-time.
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
import random
import math

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.models.rl.base_agent import NQRLAgent
from nq_alpha_elite.models.technical.indicators import TechnicalIndicators

# Configure logging
logger = logging.getLogger("NQAlpha.ContinuousLearning")

class NQLiveTrainer:
    """
    Live Trainer for NQ Alpha Elite

    This class provides continuous learning capabilities for the trading system,
    allowing it to learn from live market data in real-time without requiring
    historical data.
    """

    def __init__(self, initial_capital=100000, config=None, logger=None):
        """
        Initialize the live trainer

        Args:
            initial_capital (float): Initial capital for paper trading
            config (dict, optional): Configuration parameters
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.ContinuousLearning")

        # Initialize configuration
        self.config = config or config.CONTINUOUS_LEARNING_CONFIG.copy()

        # Initialize state
        self.running = False
        self.thread = None
        self.market_data_feed = None
        self.data_points_collected = 0
        self.training_count = 0
        self.last_training_time = None

        # Initialize paper trading
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = []
        self.trades_history = []

        # Initialize RL agent
        self.rl_agent = NQRLAgent(state_size=20, action_size=3)

        # Initialize technical indicators
        self.indicators = TechnicalIndicators()

        # Initialize market data
        self.market_data = pd.DataFrame()

        # Initialize metrics
        self.metrics = {
            'start_time': None,
            'data_points': 0,
            'training_iterations': 0,
            'trades': 0,
            'win_rate': 0.0,
            'return': 0.0,
            'drawdown': 0.0
        }

        self.logger.info(f"NQ Live Trainer initialized with {initial_capital:.2f} capital")

    def start(self):
        """Start the continuous learning system"""
        if self.running:
            self.logger.warning("Continuous learning system already running")
            return

        self.logger.info("Starting continuous learning system")

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
                target=self._trainer_thread,
                name="LiveTrainerThread"
            )
            self.thread.daemon = True
            self.thread.start()

            self.logger.info("Continuous learning thread started")

        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting continuous learning system: {e}")

    def stop(self):
        """Stop the continuous learning system"""
        if not self.running:
            self.logger.warning("Continuous learning system not running")
            return

        self.logger.info("Stopping continuous learning system")

        try:
            # Set running flag
            self.running = False

            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)

            # Close all positions
            self._close_all_positions()

            # Save model
            self._save_model()

            self.logger.info("Continuous learning system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping continuous learning system: {e}")

    def _trainer_thread(self):
        """Background thread for continuous learning"""
        self.logger.info("Continuous learning thread running")

        try:
            # Initialize variables
            last_update_time = time.time()
            last_print_time = time.time()
            last_training_time = time.time()

            while self.running:
                try:
                    # Get current time
                    current_time = time.time()

                    # Check if it's time to update
                    if current_time - last_update_time >= self.config['update_interval']:
                        # Update market data
                        self._update_market_data()

                        # Update last update time
                        last_update_time = current_time

                    # Check if it's time to train
                    if self.data_points_collected >= self.config['min_data_points'] and current_time - last_training_time >= 60:
                        # Train the agent
                        self._train_agent()

                        # Update last training time
                        last_training_time = current_time

                    # Check if it's time to print metrics
                    if current_time - last_print_time >= 60:
                        # Print metrics
                        self._print_metrics()

                        # Update last print time
                        last_print_time = current_time

                    # Sleep for a short time
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"Error in continuous learning loop: {e}")
                    time.sleep(1.0)

        except Exception as e:
            self.logger.error(f"Fatal error in continuous learning thread: {e}")

        self.logger.info("Continuous learning thread stopped")

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

            # Convert to DataFrame
            latest_df = pd.DataFrame([latest_data])

            # Add to market data
            if self.market_data.empty:
                self.market_data = latest_df
            else:
                self.market_data = pd.concat([self.market_data, latest_df], ignore_index=True)

            # Update data points collected
            self.data_points_collected = len(self.market_data)

            # Update metrics
            self.metrics['data_points'] = self.data_points_collected

            # Check if we have enough data for trading
            if self.data_points_collected >= self.config['min_data_points']:
                # Add technical indicators
                market_data_with_indicators = self.indicators.add_indicators(self.market_data)

                # Execute trading strategy
                self._execute_trading_strategy(market_data_with_indicators)

            self.logger.debug(f"Updated market data: {self.data_points_collected} points")

        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")

    def _train_agent(self):
        """Train the RL agent on collected market data"""
        try:
            # Check if we have enough data
            if self.data_points_collected < self.config['min_data_points']:
                self.logger.debug(f"Not enough data for training: {self.data_points_collected} < {self.config['min_data_points']}")
                return

            # Add technical indicators
            market_data_with_indicators = self.indicators.add_indicators(self.market_data)

            # Train the agent
            training_result = self.rl_agent.train(market_data_with_indicators, episodes=10, batch_size=32)

            if training_result:
                # Update training count
                self.training_count += 1
                self.last_training_time = datetime.now()

                # Update metrics
                self.metrics['training_iterations'] = self.training_count

                self.logger.info(f"Trained RL agent: {self.training_count} iterations")

        except Exception as e:
            self.logger.error(f"Error training agent: {e}")

    def _execute_trading_strategy(self, market_data):
        """Execute trading strategy based on RL agent and technical indicators"""
        try:
            # Check if we have enough data
            if len(market_data) < 30:
                return

            # Get latest data
            latest = market_data.iloc[-1]

            # Extract features for RL state
            state = self._extract_features(market_data)

            # Get action from RL agent
            action = self.rl_agent.act(state)

            # Execute action
            self._execute_action(action, latest)

        except Exception as e:
            self.logger.error(f"Error executing trading strategy: {e}")

    def _extract_features(self, market_data):
        """Extract features for RL state"""
        try:
            # Use last 30 data points
            recent_data = market_data.iloc[-30:]

            # Create feature vector
            features = []

            # Price features
            if 'Close' in recent_data.columns:
                close_prices = recent_data['Close'].values
            elif 'price' in recent_data.columns:
                close_prices = recent_data['price'].values
            else:
                close_prices = np.zeros(30)

            # Normalize prices
            if len(close_prices) > 0 and close_prices[0] != 0:
                normalized_close = close_prices / close_prices[0] - 1
            else:
                normalized_close = np.zeros_like(close_prices)

            features.extend(normalized_close)

            # Technical indicators
            if 'RSI' in recent_data.columns:
                features.append(recent_data['RSI'].iloc[-1] / 100)

            if 'MACD' in recent_data.columns:
                features.append(recent_data['MACD'].iloc[-1] / 100)

            # Ensure we have the right state size
            if len(features) < self.rl_agent.state_size:
                features.extend([0] * (self.rl_agent.state_size - len(features)))
            elif len(features) > self.rl_agent.state_size:
                features = features[:self.rl_agent.state_size]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(self.rl_agent.state_size, dtype=np.float32)

    def _execute_action(self, action, latest_data):
        """
        Execute trading action

        Args:
            action (int): Action from RL agent (0: buy, 1: sell, 2: hold)
            latest_data (Series): Latest market data
        """
        try:
            # Get current price
            if 'Close' in latest_data:
                current_price = latest_data['Close']
            elif 'price' in latest_data:
                current_price = latest_data['price']
            else:
                self.logger.warning("No price data available")
                return

            # Get current time
            current_time = datetime.now()

            # Calculate position size based on data points collected
            confidence = min(1.0, self.data_points_collected / self.config['preferred_data_points'])
            position_size = self.config['initial_trade_size_pct'] + (self.config['max_trade_size_pct'] - self.config['initial_trade_size_pct']) * confidence

            # Calculate quantity
            quantity = math.floor(self.capital * position_size / current_price)

            # Execute action
            if action == 0:  # Buy
                # Check if we already have a long position
                if any(p['side'] == 'long' for p in self.positions):
                    self.logger.debug("Already have a long position, skipping buy")
                    return

                # Close any short positions
                self._close_positions('short')

                # Open long position
                if quantity > 0:
                    position = {
                        'side': 'long',
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'current_price': current_price,
                        'pnl': 0.0
                    }

                    self.positions.append(position)

                    self.logger.info(f"Opened long position: {quantity} @ {current_price:.2f}")

            elif action == 1:  # Sell
                # Check if we already have a short position
                if any(p['side'] == 'short' for p in self.positions):
                    self.logger.debug("Already have a short position, skipping sell")
                    return

                # Close any long positions
                self._close_positions('long')

                # Open short position
                if quantity > 0:
                    position = {
                        'side': 'short',
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'current_price': current_price,
                        'pnl': 0.0
                    }

                    self.positions.append(position)

                    self.logger.info(f"Opened short position: {quantity} @ {current_price:.2f}")

            # Update metrics
            self.metrics['trades'] = len(self.trades_history)

        except Exception as e:
            self.logger.error(f"Error executing action: {e}")

    def _update_positions(self, current_price):
        """
        Update positions with current price

        Args:
            current_price (float): Current price
        """
        try:
            # Update each position
            for position in self.positions:
                # Update current price
                position['current_price'] = current_price

                # Calculate P&L
                if position['side'] == 'long':
                    position['pnl'] = (current_price - position['entry_price']) * position['quantity']
                else:
                    position['pnl'] = (position['entry_price'] - current_price) * position['quantity']

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

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

                # Update capital
                self.capital += pnl

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

            # Calculate return
            self.metrics['return'] = (self.capital / self.initial_capital - 1) * 100

        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    def _close_all_positions(self):
        """Close all positions"""
        self._close_positions()

    def _save_model(self):
        """Save the RL agent model"""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save model
            save_path = os.path.join(config.MODELS_DIR, "rl", f"live_trained_{timestamp}")

            # Save agent
            self.rl_agent.save(save_path)

            self.logger.info(f"Saved RL agent model to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def _print_metrics(self):
        """Print performance metrics"""
        try:
            # Calculate metrics
            runtime = datetime.now() - self.metrics['start_time'] if self.metrics['start_time'] else timedelta(0)
            runtime_str = str(runtime).split('.')[0]  # Remove microseconds

            # Print metrics
            print("\n" + "=" * 50)
            print(f"  CONTINUOUS LEARNING METRICS")
            print("=" * 50)
            print(f"Runtime: {runtime_str}")
            print(f"Data Points: {self.data_points_collected}")
            print(f"Training Iterations: {self.training_count}")
            print(f"Trades: {len(self.trades_history)}")
            print(f"Win Rate: {self.metrics['win_rate']:.2%}")
            print(f"Return: {self.metrics['return']:.2f}%")
            print(f"Current Capital: ${self.capital:.2f}")

            # Print active positions
            if self.positions:
                print("\nActive Positions:")
                for i, position in enumerate(self.positions):
                    print(f"  {i+1}. {position['side'].upper()}: {position['quantity']} @ {position['entry_price']:.2f}, P&L: {position['pnl']:.2f}")

            print("=" * 50)

        except Exception as e:
            self.logger.error(f"Error printing metrics: {e}")

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

            # Calculate drawdown
            if self.trades_history:
                # Calculate running balance
                balances = [self.initial_capital]
                for trade in self.trades_history:
                    balances.append(balances[-1] + trade['pnl'])

                # Calculate drawdown
                peak = self.initial_capital
                drawdowns = []

                for balance in balances:
                    if balance > peak:
                        peak = balance

                    drawdown = (peak - balance) / peak * 100
                    drawdowns.append(drawdown)

                max_drawdown = max(drawdowns)
            else:
                max_drawdown = 0.0

            # Create summary
            summary = {
                'runtime': runtime_str,
                'data_points': self.data_points_collected,
                'training_iterations': self.training_count,
                'trades': len(self.trades_history),
                'win_rate': self.metrics['win_rate'],
                'return': self.metrics['return'],
                'max_drawdown': max_drawdown,
                'current_capital': self.capital,
                'active_positions': len(self.positions)
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
