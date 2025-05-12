#!/usr/bin/env python3
"""
Configuration settings for NQ Alpha Elite Trading System
"""
import os
import logging
from datetime import datetime

# System version
VERSION = "4.0"
VERSION_NAME = "Quantum Intelligence"
DEVELOPER = "An0nym0usn3thunt3r"

# Paths and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "market_data"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "accumulated"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "paper_trades"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "rl"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "a3c"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Market data configuration
MARKET_DATA_CONFIG = {
    'symbol': 'NQ',  # Futures symbol
    'update_interval': 2.0,  # Update interval in seconds
    'data_dir': os.path.join(DATA_DIR, "market_data"),  # Directory for data storage
    'current_contract': 'NQM25',  # Current NQ futures contract (June 2025)
    'timeout': 15,  # Request timeout
    'retry_count': 3,  # Number of retries
    'min_price': 18000,  # Minimum reasonable price
    'max_price': 22000,  # Maximum reasonable price
    'max_data_points': 10000,  # Maximum data points to store
    'compress_old_data': True,  # Compress older data for efficiency
    'data_acceleration': False,  # Generate synthetic ticks
    'synthetic_volatility': 0.0001,  # Volatility for synthetic data
    'headers': {  # Request headers
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    },
    'debug_requests': False  # Debug HTTP requests
}

# Reinforcement Learning configuration
RL_CONFIG = {
    'algorithm': 'ppo',  # Default algorithm (ppo, a2c, dqn)
    'state_size': 30,  # State representation size
    'action_size': 3,  # Number of actions (buy, sell, hold)
    'learning_rate': 0.0001,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'batch_size': 64,  # Batch size for training
    'memory_size': 10000,  # Replay memory size
    'target_update': 100,  # Target network update frequency
    'save_dir': os.path.join(MODELS_DIR, "rl"),  # Directory to save models
    'load_latest': True,  # Load latest model if available
}

# Trading parameters
TRADING_CONFIG = {
    'default_symbol': 'NQ',  # Default trading symbol
    'default_timeframe': '1h',  # Default timeframe
    'default_initial_balance': 10000,  # Default initial balance for paper trading
    'default_position_size': 0.1,  # Default position size (10% of capital)
    'max_position_size': 0.5,  # Maximum position size (50% of capital)
    'use_stop_loss': True,  # Use stop loss
    'stop_loss_pct': 0.02,  # Stop loss percentage (2%)
    'use_take_profit': True,  # Use take profit
    'take_profit_pct': 0.03,  # Take profit percentage (3%)
    'max_trades_per_day': 5,  # Maximum trades per day
}

# Continuous learning configuration
CONTINUOUS_LEARNING_CONFIG = {
    'update_interval': 5.0,  # Update interval in seconds
    'min_data_points': 10,  # Minimum data points before trading
    'preferred_data_points': 100,  # Preferred number of data points
    'initial_trade_size_pct': 0.01,  # Initial position size (1%)
    'max_trade_size_pct': 0.05,  # Maximum position size (5%)
    'print_metrics_interval': 20,  # Print metrics every 20 data points
}

# Technical indicators configuration
INDICATORS_CONFIG = {
    'rsi_period': 14,  # RSI period
    'macd_fast': 12,  # MACD fast period
    'macd_slow': 26,  # MACD slow period
    'macd_signal': 9,  # MACD signal period
    'bb_period': 20,  # Bollinger Bands period
    'bb_std': 2,  # Bollinger Bands standard deviation
    'atr_period': 14,  # ATR period
}

# Performance metrics configuration
METRICS_CONFIG = {
    'benchmark': 'NQ',  # Benchmark symbol
    'risk_free_rate': 0.02,  # Risk-free rate for Sharpe ratio
    'calculate_drawdown': True,  # Calculate drawdown
    'calculate_sharpe': True,  # Calculate Sharpe ratio
    'calculate_sortino': True,  # Calculate Sortino ratio
    'calculate_calmar': True,  # Calculate Calmar ratio
}
