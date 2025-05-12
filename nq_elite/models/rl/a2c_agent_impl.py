#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) Agent Implementation for NQ Alpha Elite

This module provides a complete implementation of the A2C algorithm for trading
NASDAQ 100 E-mini futures. It includes:
1. Actor-Critic network architecture
2. Advantage calculation
3. Policy gradient updates
4. Entropy regularization
5. Worker threads for parallel training
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import random
import math
import threading
import joblib

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.RL.A2C")

class NQA3CAgent:
    """
    A3C (Asynchronous Advantage Actor-Critic) Agent for NQ Alpha Elite
    
    This class implements the A3C algorithm for trading NASDAQ 100 E-mini futures.
    It uses multiple worker threads to train the agent in parallel.
    """
    
    def __init__(self, state_size=None, action_size=3, logger=None):
        """
        Initialize the A3C agent
        
        Args:
            state_size (int, optional): Dimension of state representation
            action_size (int): Number of possible actions (default: 3 - buy, sell, hold)
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.RL.A3C")
        
        # Set state and action dimensions
        self.state_size = state_size or config.RL_CONFIG['state_size']
        self.action_size = action_size
        
        # Initialize hyperparameters
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.0001  # Learning rate
        self.entropy_beta = 0.01  # Entropy regularization coefficient
        
        # Initialize models
        self.global_model = None
        self.workers = []
        self.optimizer = None
        
        # Initialize feature preprocessing
        self.feature_means = None
        self.feature_stds = None
        
        # Initialize performance tracking
        self.performance = {
            'cumulative_reward': 0.0,
            'rewards': [],
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'train_loss': []
        }
        
        # Configuration
        self.config = {
            'save_dir': config.RL_CONFIG['save_dir'],
            'load_latest': config.RL_CONFIG['load_latest'],
            'num_workers': 4  # Number of worker threads
        }
        
        # Ensure save directory exists
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info(f"NQA3CAgent initialized with state_size={self.state_size}, action_size={self.action_size}")
    
    def _initialize_models(self):
        """Initialize A3C models"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            # Create global model
            input_layer = Input(shape=(self.state_size,))
            x = Dense(128, activation='relu')(input_layer)
            x = BatchNormalization()(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Actor output (policy)
            actor_output = Dense(self.action_size, activation='softmax', name='policy')(x)
            
            # Critic output (value)
            critic_output = Dense(1, name='value')(x)
            
            # Create model
            self.global_model = Model(inputs=input_layer, outputs=[actor_output, critic_output])
            
            # Create optimizer
            self.optimizer = Adam(learning_rate=self.learning_rate)
            
            self.logger.info("A3C global model initialized")
            
        except ImportError:
            self.logger.error("TensorFlow not available, A3C requires TensorFlow")
            raise ImportError("TensorFlow is required for A3C")
    
    def _create_worker_model(self):
        """Create a worker model that shares weights with the global model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
            
            # Create worker model with same architecture as global model
            input_layer = Input(shape=(self.state_size,))
            x = Dense(128, activation='relu')(input_layer)
            x = BatchNormalization()(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Actor output (policy)
            actor_output = Dense(self.action_size, activation='softmax', name='policy')(x)
            
            # Critic output (value)
            critic_output = Dense(1, name='value')(x)
            
            # Create model
            worker_model = Model(inputs=input_layer, outputs=[actor_output, critic_output])
            
            # Copy weights from global model
            worker_model.set_weights(self.global_model.get_weights())
            
            return worker_model
            
        except ImportError:
            self.logger.error("TensorFlow not available")
            return None
    
    def _preprocess_state(self, state):
        """Preprocess state for model input
        
        Args:
            state: Raw state vector
            
        Returns:
            numpy.ndarray: Preprocessed state
        """
        try:
            # Ensure state is numpy array
            state_array = np.array(state, dtype=np.float32)
            
            # Initialize means and stds if not already done
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = np.zeros_like(state_array)
                self.feature_stds = np.ones_like(state_array)
            
            # Normalize state
            normalized_state = (state_array - self.feature_means) / self.feature_stds
            
            # Replace NaN and inf values
            normalized_state = np.nan_to_num(normalized_state, nan=0.0, posinf=0.0, neginf=0.0)
            
            return normalized_state
            
        except Exception as e:
            self.logger.error(f"Error preprocessing state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
    
    def act(self, state, training=False):
        """Select action based on current state
        
        Args:
            state: Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        try:
            import tensorflow as tf
            
            # Preprocess state
            processed_state = self._preprocess_state(state)
            
            # Reshape for model input
            processed_state = np.reshape(processed_state, [1, self.state_size])
            
            # Get policy and value from model
            policy, value = self.global_model.predict(processed_state, verbose=0)
            
            # Get action probabilities
            action_probs = policy[0]
            
            if training:
                # Sample action from policy
                action = np.random.choice(self.action_size, p=action_probs)
            else:
                # Use deterministic policy for evaluation
                action = np.argmax(action_probs)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error selecting action: {e}")
            # Default to hold action (2) in case of error
            return 2
    
    def train(self, market_data=None, episodes=100, batch_size=None):
        """Train the A3C agent
        
        Args:
            market_data (DataFrame, optional): Market data for training
            episodes (int): Number of training episodes
            batch_size (int, optional): Batch size for training (not used in A3C)
            
        Returns:
            dict: Training metrics
        """
        try:
            import tensorflow as tf
            
            # Check if we have market data
            if market_data is None or len(market_data) < 100:
                self.logger.warning("Insufficient market data for training")
                return None
            
            # Extract features from market data
            states = self._extract_features(market_data)
            
            if not states:
                self.logger.warning("No valid states extracted from market data")
                return None
            
            # Fit preprocessor
            self._fit_preprocessor(states)
            
            # Create worker threads
            self._create_workers(market_data, states, episodes)
            
            # Start worker threads
            self._start_workers()
            
            # Wait for workers to complete
            self._wait_for_workers()
            
            # Return training metrics
            return {
                'episodes': episodes,
                'final_loss': self.performance['train_loss'][-1] if self.performance['train_loss'] else None,
                'mean_loss': np.mean(self.performance['train_loss']) if self.performance['train_loss'] else None,
                'cumulative_reward': self.performance['cumulative_reward']
            }
            
        except Exception as e:
            self.logger.error(f"Error training agent: {e}")
            self.logger.error(traceback.format_exc())
            return None
