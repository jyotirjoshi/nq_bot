#!/usr/bin/env python3
"""
Base Reinforcement Learning Agent for NQ Alpha Elite

This module provides the base class for all RL agents in the system,
with common functionality for state representation, action selection,
and model management.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import traceback
from collections import deque
import random

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.RL.BaseAgent")

class NQRLAgent:
    """
    Base Reinforcement Learning Agent for NQ Alpha Elite

    This class provides common functionality for all RL agents in the system,
    including state representation, action selection, and model management.
    """

    def __init__(self, state_size=None, action_size=3, model_type='tensorflow', logger=None):
        """
        Initialize the RL agent

        Args:
            state_size (int, optional): Dimension of state representation
            action_size (int): Number of possible actions (default: 3 - buy, sell, hold)
            model_type (str): Type of model to use ('tensorflow' or 'sklearn')
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.RL.BaseAgent")

        # Set state and action dimensions
        self.state_size = state_size or config.RL_CONFIG['state_size']
        self.action_size = action_size
        self.model_type = model_type

        # Initialize hyperparameters
        self.gamma = config.RL_CONFIG['gamma']  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = config.RL_CONFIG['learning_rate']

        # Initialize memory
        self.memory_size = config.RL_CONFIG['memory_size']
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = config.RL_CONFIG['batch_size']

        # Initialize training parameters
        self.warmup_steps = 1000  # Steps before training starts
        self.train_interval = 4  # Train every N steps
        self.target_update = config.RL_CONFIG['target_update']  # Update target network every N steps
        self.step_count = 0
        self.train_count = 0

        # Initialize models
        self.main_model = None
        self.target_model = None

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
            'load_latest': config.RL_CONFIG['load_latest']
        }

        # Ensure save directory exists
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # Initialize models
        self._initialize_models()

        self.logger.info(f"NQRLAgent initialized with state_size={self.state_size}, action_size={self.action_size}")

    def _initialize_models(self):
        """Initialize RL models based on model type"""
        if self.model_type == 'tensorflow':
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
                from tensorflow.keras.optimizers import Adam

                # Create main model
                self.main_model = Sequential([
                    Dense(128, input_dim=self.state_size, activation='relu'),
                    BatchNormalization(),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(self.action_size, activation='linear')
                ])

                # Compile model
                self.main_model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss='mse'
                )

                # Create target model (for stability)
                self.target_model = Sequential([
                    Dense(128, input_dim=self.state_size, activation='relu'),
                    BatchNormalization(),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(self.action_size, activation='linear')
                ])

                # Copy weights from main model to target model
                self.target_model.set_weights(self.main_model.get_weights())

                self.logger.info("TensorFlow models initialized")

            except ImportError:
                self.logger.warning("TensorFlow not available, falling back to sklearn")
                self.model_type = 'sklearn'
                self._initialize_sklearn_model()

        else:
            self._initialize_sklearn_model()

    def _initialize_sklearn_model(self):
        """Initialize sklearn model as fallback"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            # Create main model
            self.main_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=self.learning_rate,
                max_depth=3,
                random_state=42
            )

            # Create target model (copy of main model)
            self.target_model = None  # Will be created after first fit

            self.logger.info("Sklearn model initialized")

        except ImportError:
            self.logger.error("Neither TensorFlow nor sklearn available")
            raise ImportError("No suitable ML library available")

    def _extract_features(self, market_data):
        """Extract features from market data for RL state representation

        Args:
            market_data: DataFrame with market data

        Returns:
            list: List of state representations
        """
        try:
            # Check if we have enough data
            if market_data is None or len(market_data) < 30:
                self.logger.warning("Insufficient market data for feature extraction")
                return []

            # Ensure we have price data
            price_col = None
            for col in ['Close', 'price', 'last_price']:
                if col in market_data.columns:
                    price_col = col
                    break

            if price_col is None:
                self.logger.error("No price column found in market data")
                return []

            # Extract features
            states = []

            # Use a sliding window approach
            window_size = 30  # Use 30 periods of data for each state

            for i in range(window_size, len(market_data)):
                # Extract window of data
                window = market_data.iloc[i-window_size:i]

                # Create feature vector
                features = []

                # Price features
                close_prices = window[price_col].values
                normalized_close = close_prices / close_prices[0] - 1  # Normalize to percentage change
                features.extend(normalized_close)

                # Technical indicators if available
                if 'RSI' in window.columns:
                    features.append(window['RSI'].iloc[-1] / 100)  # Normalize RSI

                if 'MACD' in window.columns:
                    # Normalize MACD
                    macd_vals = window['MACD'].values
                    macd_std = np.std(macd_vals) if np.std(macd_vals) > 0 else 1
                    features.append(window['MACD'].iloc[-1] / macd_std)

                # Volatility
                returns = np.diff(close_prices) / close_prices[:-1]
                features.append(np.std(returns))

                # Volume features if available
                if 'Volume' in window.columns or 'volume' in window.columns:
                    vol_col = 'Volume' if 'Volume' in window.columns else 'volume'
                    volume = window[vol_col].values
                    normalized_volume = volume / np.mean(volume) if np.mean(volume) > 0 else volume
                    features.append(normalized_volume[-1])

                # Order flow features if available
                if 'delta' in window.columns:
                    features.append(window['delta'].iloc[-1])

                if 'order_flow' in window.columns:
                    features.append(window['order_flow'].iloc[-1])

                # Ensure we have the right state size
                features = self._pad_feature_vector(features)

                states.append(np.array(features, dtype=np.float32))

            return states

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _fit_preprocessor(self, states):
        """Fit feature preprocessor for normalization

        Args:
            states: List of state vectors
        """
        try:
            # Convert to numpy array
            states_array = np.array(states)

            # Calculate mean and std for each feature
            self.feature_means = np.mean(states_array, axis=0)
            self.feature_stds = np.std(states_array, axis=0)

            # Replace zero std with 1 to avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1.0

            self.logger.debug("Feature preprocessor fitted")

        except Exception as e:
            self.logger.error(f"Error fitting preprocessor: {e}")
            self.feature_means = np.zeros(self.state_size)
            self.feature_stds = np.ones(self.state_size)

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

    def _pad_feature_vector(self, features):
        """Pad or truncate feature vector to match state_size

        Args:
            features: List of features

        Returns:
            list: Padded/truncated feature vector
        """
        # Ensure we have the right state size
        if len(features) < self.state_size:
            # Pad with zeros
            features.extend([0] * (self.state_size - len(features)))
        elif len(features) > self.state_size:
            # Truncate
            features = features[:self.state_size]

        return features

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            # Preprocess states
            processed_state = self._preprocess_state(state)
            processed_next_state = self._preprocess_state(next_state)

            # Add to memory
            self.memory.append((processed_state, action, reward, processed_next_state, done))

            # Update performance tracking
            self.performance['cumulative_reward'] += reward
            self.performance['rewards'].append(reward)

            if done:
                # Track trade outcome
                self.performance['trades'] += 1
                if reward > 0:
                    self.performance['wins'] += 1
                else:
                    self.performance['losses'] += 1

                # Update win rate
                if self.performance['trades'] > 0:
                    self.performance['win_rate'] = self.performance['wins'] / self.performance['trades']

        except Exception as e:
            self.logger.error(f"Error storing experience: {e}")

    def act(self, state, training=False):
        """Select action based on current state

        Args:
            state: Current state
            training (bool): Whether in training mode (enables exploration)

        Returns:
            int: Selected action
        """
        try:
            # Preprocess state
            processed_state = self._preprocess_state(state)

            # Reshape for model input
            if self.model_type == 'tensorflow':
                processed_state = np.reshape(processed_state, [1, self.state_size])

            # Exploration (epsilon-greedy)
            if training and np.random.rand() <= self.epsilon:
                # Random action
                return random.randrange(self.action_size)

            # Exploitation (use model)
            if self.model_type == 'tensorflow':
                # Use TensorFlow model
                q_values = self.main_model.predict(processed_state, verbose=0)[0]
                return np.argmax(q_values)
            else:
                # Use sklearn model
                if not hasattr(self.main_model, 'predict'):
                    # Model not trained yet
                    return random.randrange(self.action_size)

                # Predict Q-value for each action
                q_values = np.array([
                    self.main_model.predict([processed_state])[0]
                    for _ in range(self.action_size)
                ])

                return np.argmax(q_values)

        except Exception as e:
            self.logger.error(f"Error selecting action: {e}")
            # Default to hold action (2) in case of error
            return 2

    def train(self, market_data=None, episodes=100, batch_size=None):
        """Train the RL agent

        Args:
            market_data (DataFrame, optional): Market data for training
            episodes (int): Number of training episodes
            batch_size (int, optional): Batch size for training

        Returns:
            dict: Training metrics
        """
        try:
            # Use configured batch size if not provided
            batch_size = batch_size or self.batch_size

            # Check if we have market data
            if market_data is not None:
                # Extract features from market data
                states = self._extract_features(market_data)

                if not states:
                    self.logger.warning("No valid states extracted from market data")
                    return None

                # Fit preprocessor
                self._fit_preprocessor(states)

                # Generate experiences from market data
                self._generate_experiences(market_data, states)

            # Check if we have enough experiences
            if len(self.memory) < batch_size:
                self.logger.warning(f"Not enough experiences for training: {len(self.memory)} < {batch_size}")
                return None

            # Training loop
            losses = []

            for episode in range(episodes):
                # Sample batch from memory
                minibatch = random.sample(self.memory, batch_size)

                if self.model_type == 'tensorflow':
                    loss = self._train_tensorflow(minibatch)
                else:
                    loss = self._train_sklearn(minibatch)

                losses.append(loss)

                # Update target model periodically
                if self.train_count % self.target_update == 0:
                    self._update_target_model()

                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                # Update training count
                self.train_count += 1

                # Log progress
                if (episode + 1) % 10 == 0:
                    self.logger.info(f"Episode {episode+1}/{episodes}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.4f}")

            # Update performance tracking
            self.performance['train_loss'].extend(losses)

            # Return training metrics
            return {
                'episodes': episodes,
                'final_loss': losses[-1] if losses else None,
                'mean_loss': np.mean(losses) if losses else None,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory)
            }

        except Exception as e:
            self.logger.error(f"Error training agent: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _train_tensorflow(self, minibatch):
        """Train TensorFlow model on minibatch

        Args:
            minibatch: Batch of experiences

        Returns:
            float: Loss value
        """
        # Extract batch components
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Calculate target Q-values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(target_q_values, axis=1)

        # Calculate target using Bellman equation
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Get current Q-values
        current_q_values = self.main_model.predict(states, verbose=0)

        # Update only the Q-values for the actions taken
        for i, action in enumerate(actions):
            current_q_values[i][action] = targets[i]

        # Train the model
        history = self.main_model.fit(
            states, current_q_values,
            epochs=1, verbose=0,
            batch_size=len(minibatch)
        )

        # Return loss
        return history.history['loss'][0]

    def _train_sklearn(self, minibatch):
        """Train sklearn model on minibatch

        Args:
            minibatch: Batch of experiences

        Returns:
            float: Loss value
        """
        # Extract batch components
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Initialize target model if not already done
        if self.target_model is None:
            from sklearn.ensemble import GradientBoostingRegressor
            self.target_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=self.learning_rate,
                max_depth=3,
                random_state=42
            )

        # Calculate target Q-values
        if hasattr(self.target_model, 'predict'):
            # Use target model if trained
            next_q_values = np.array([
                self.target_model.predict(next_states)
                for _ in range(self.action_size)
            ]).T
            max_next_q_values = np.max(next_q_values, axis=1)
        else:
            # Use random values if target model not trained
            max_next_q_values = np.zeros(len(minibatch))

        # Calculate target using Bellman equation
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Train the model
        self.main_model.fit(states, targets)

        # Copy main model to target model
        self.target_model = joblib.clone(self.main_model)

        # Calculate loss (MSE)
        predictions = self.main_model.predict(states)
        loss = np.mean((predictions - targets) ** 2)

        return loss

    def _update_target_model(self):
        """Update target model with weights from main model"""
        if self.model_type == 'tensorflow':
            # Copy weights from main model to target model
            self.target_model.set_weights(self.main_model.get_weights())
        else:
            # Clone main model to target model
            self.target_model = joblib.clone(self.main_model)

        self.logger.debug("Target model updated")

    def _generate_experiences(self, market_data, states):
        """Generate experiences from market data for training

        Args:
            market_data: DataFrame with market data
            states: List of state representations
        """
        try:
            # Check if we have enough data
            if len(states) < 2:
                self.logger.warning("Insufficient data for experience generation")
                return

            # Get price column
            price_col = None
            for col in ['Close', 'price', 'last_price']:
                if col in market_data.columns:
                    price_col = col
                    break

            if price_col is None:
                self.logger.error("No price column found in market data")
                return

            # Generate experiences
            for i in range(len(states) - 1):
                # Get current and next state
                state = states[i]
                next_state = states[i + 1]

                # Get current and next price
                current_price = market_data[price_col].iloc[i + 30]  # Offset by window size
                next_price = market_data[price_col].iloc[i + 31]  # Next price

                # Try all actions and calculate rewards
                for action in range(self.action_size):
                    # Calculate reward based on action and price movement
                    if action == 0:  # Buy
                        reward = (next_price - current_price) / current_price
                    elif action == 1:  # Sell
                        reward = (current_price - next_price) / current_price
                    else:  # Hold
                        reward = 0.0001  # Small positive reward for holding

                    # Scale reward for better learning
                    reward *= 100  # Convert to percentage

                    # Determine if episode is done
                    done = (i == len(states) - 2)

                    # Store experience
                    self.remember(state, action, reward, next_state, done)

            self.logger.info(f"Generated {len(states) - 1} experiences from market data")

        except Exception as e:
            self.logger.error(f"Error generating experiences: {e}")
            self.logger.error(traceback.format_exc())

    def save(self, path=None):
        """Save agent to disk

        Args:
            path (str, optional): Path to save agent

        Returns:
            bool: Success status
        """
        try:
            # Use default path if not provided
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.config['save_dir'], f"agent_{timestamp}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save models
            if self.model_type == 'tensorflow':
                # Save TensorFlow models
                self.main_model.save(f"{path}_main.h5")
                self.target_model.save(f"{path}_target.h5")
            else:
                # Save sklearn models
                joblib.dump(self.main_model, f"{path}_main.joblib")
                if self.target_model is not None:
                    joblib.dump(self.target_model, f"{path}_target.joblib")

            # Save agent state
            agent_state = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'model_type': self.model_type,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'performance': self.performance,
                'train_count': self.train_count
            }

            joblib.dump(agent_state, f"{path}_state.joblib")

            self.logger.info(f"Agent saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving agent: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def load(self, path):
        """Load agent from disk

        Args:
            path (str): Path to load agent from

        Returns:
            bool: Success status
        """
        try:
            # Check if state file exists
            state_path = f"{path}_state.joblib"
            if not os.path.exists(state_path):
                self.logger.error(f"Agent state file not found: {state_path}")
                return False

            # Load agent state
            agent_state = joblib.load(state_path)

            # Update agent parameters
            self.state_size = agent_state['state_size']
            self.action_size = agent_state['action_size']
            self.model_type = agent_state['model_type']
            self.gamma = agent_state['gamma']
            self.epsilon = agent_state['epsilon']
            self.epsilon_min = agent_state['epsilon_min']
            self.epsilon_decay = agent_state['epsilon_decay']
            self.learning_rate = agent_state['learning_rate']
            self.feature_means = agent_state['feature_means']
            self.feature_stds = agent_state['feature_stds']
            self.performance = agent_state['performance']
            self.train_count = agent_state['train_count']

            # Load models
            if self.model_type == 'tensorflow':
                # Load TensorFlow models
                try:
                    import tensorflow as tf
                    self.main_model = tf.keras.models.load_model(f"{path}_main.h5")
                    self.target_model = tf.keras.models.load_model(f"{path}_target.h5")
                except ImportError:
                    self.logger.error("TensorFlow not available")
                    return False
            else:
                # Load sklearn models
                self.main_model = joblib.load(f"{path}_main.joblib")
                target_path = f"{path}_target.joblib"
                if os.path.exists(target_path):
                    self.target_model = joblib.load(target_path)

            self.logger.info(f"Agent loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading agent: {e}")
            self.logger.error(traceback.format_exc())
            return False
