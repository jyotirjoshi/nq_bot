#!/usr/bin/env python3
"""
A3C Worker Implementation for NQ Alpha Elite

This module provides the worker thread implementation for the A3C algorithm.
Each worker has its own copy of the model and environment, and updates the
global model asynchronously.
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

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.RL.A3C.Worker")

class A3CWorker:
    """
    Worker thread for A3C algorithm

    Each worker has its own copy of the model and environment,
    and updates the global model asynchronously.
    """

    def __init__(self, name, global_model, optimizer, market_data, states, episodes,
                 state_size, action_size, gamma, entropy_beta, feature_means, feature_stds, logger=None):
        """
        Initialize A3C worker

        Args:
            name (str): Worker name
            global_model: Global model to update
            optimizer: Optimizer for model updates
            market_data: DataFrame with market data
            states: List of state representations
            episodes (int): Number of episodes to train
            state_size (int): Dimension of state representation
            action_size (int): Number of possible actions
            gamma (float): Discount factor
            entropy_beta (float): Entropy regularization coefficient
            feature_means: Mean values for feature normalization
            feature_stds: Standard deviation values for feature normalization
            logger (logging.Logger, optional): Logger instance
        """
        self.name = name
        self.global_model = global_model
        self.optimizer = optimizer
        self.market_data = market_data
        self.states = states
        self.episodes = episodes
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.logger = logger or logging.getLogger(f"NQAlpha.RL.A3C.{name}")

        # Create local model
        self.local_model = self._create_local_model()

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

        self.logger.info(f"A3C worker {name} initialized")

    def _create_local_model(self):
        """Create local model with same architecture as global model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization

            # Create local model with same architecture as global model
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
            local_model = Model(inputs=input_layer, outputs=[actor_output, critic_output])

            # Copy weights from global model
            local_model.set_weights(self.global_model.get_weights())

            return local_model

        except Exception as e:
            self.logger.error(f"Error creating local model: {e}")
            return None

    def run(self):
        """Run worker training loop"""
        try:
            import tensorflow as tf

            # Get price column
            price_col = None
            for col in ['Close', 'price', 'last_price']:
                if col in self.market_data.columns:
                    price_col = col
                    break

            if price_col is None:
                self.logger.error("No price column found in market data")
                return

            # Training loop
            for episode in range(self.episodes):
                # Reset gradients
                with tf.GradientTape() as tape:
                    # Initialize episode variables
                    memory = []
                    episode_reward = 0
                    episode_steps = 0

                    # Get random starting point
                    start_idx = random.randint(0, len(self.states) - 100)

                    # Get initial state
                    state = self.states[start_idx]

                    # Episode loop
                    done = False
                    while not done and episode_steps < 100:
                        # Preprocess state
                        processed_state = self._preprocess_state(state)
                        processed_state = np.reshape(processed_state, [1, self.state_size])

                        # Get policy and value from local model
                        policy, value = self.local_model.predict(processed_state, verbose=0)

                        # Get action probabilities
                        action_probs = policy[0]

                        # Sample action from policy
                        action = np.random.choice(self.action_size, p=action_probs)

                        # Get current and next price
                        current_idx = start_idx + episode_steps
                        if current_idx + 1 >= len(self.market_data):
                            done = True
                            continue

                        current_price = self.market_data[price_col].iloc[current_idx]
                        next_price = self.market_data[price_col].iloc[current_idx + 1]

                        # Calculate reward based on action and price movement
                        if action == 0:  # Buy
                            reward = (next_price - current_price) / current_price
                        elif action == 1:  # Sell
                            reward = (current_price - next_price) / current_price
                        else:  # Hold
                            reward = 0.0001  # Small positive reward for holding

                        # Scale reward for better learning
                        reward *= 100  # Convert to percentage

                        # Get next state
                        if current_idx + 1 < len(self.states):
                            next_state = self.states[current_idx + 1]
                        else:
                            next_state = state  # Use current state if at end of data
                            done = True

                        # Store experience
                        memory.append((state, action, reward, next_state, done, action_probs, value[0][0]))

                        # Update state
                        state = next_state

                        # Update episode variables
                        episode_reward += reward
                        episode_steps += 1

                        # Check if episode is done
                        if episode_steps >= 100:
                            done = True

                    # Update performance tracking
                    self.performance['cumulative_reward'] += episode_reward
                    self.performance['rewards'].append(episode_reward)

                    if episode_reward > 0:
                        self.performance['wins'] += 1
                    else:
                        self.performance['losses'] += 1

                    self.performance['trades'] += 1

                    if self.performance['trades'] > 0:
                        self.performance['win_rate'] = self.performance['wins'] / self.performance['trades']

                    # Calculate gradients and update global model
                    loss = self._update_global_model(memory)

                    # Store loss
                    self.performance['train_loss'].append(loss)

                    # Log progress
                    if (episode + 1) % 10 == 0:
                        self.logger.info(f"Worker {self.name} - Episode {episode+1}/{self.episodes}, "
                                        f"Reward: {episode_reward:.4f}, Loss: {loss:.4f}")

            self.logger.info(f"Worker {self.name} completed training")

        except Exception as e:
            self.logger.error(f"Error in worker {self.name}: {e}")
            self.logger.error(traceback.format_exc())

    def _preprocess_state(self, state):
        """Preprocess state for model input"""
        try:
            # Ensure state is numpy array
            state_array = np.array(state, dtype=np.float32)

            # Normalize state
            normalized_state = (state_array - self.feature_means) / self.feature_stds

            # Replace NaN and inf values
            normalized_state = np.nan_to_num(normalized_state, nan=0.0, posinf=0.0, neginf=0.0)

            return normalized_state

        except Exception as e:
            self.logger.error(f"Error preprocessing state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)

    def _update_global_model(self, memory):
        """Update global model using experiences from memory"""
        try:
            import tensorflow as tf

            # Check if we have experiences
            if not memory:
                return 0.0

            # Calculate returns and advantages
            returns = []
            advantages = []

            # Get last value
            last_state = memory[-1][3]
            processed_state = self._preprocess_state(last_state)
            processed_state = np.reshape(processed_state, [1, self.state_size])
            _, last_value = self.local_model.predict(processed_state, verbose=0)
            last_value = last_value[0][0]

            # Initialize return with last value
            R = last_value if not memory[-1][4] else 0

            # Calculate returns and advantages in reverse order
            for state, action, reward, next_state, done, action_probs, value in reversed(memory):
                R = reward + self.gamma * R * (1 - done)
                advantage = R - value

                returns.append(R)
                advantages.append(advantage)

            # Reverse lists
            returns = returns[::-1]
            advantages = advantages[::-1]

            # Convert to numpy arrays
            returns = np.array(returns, dtype=np.float32)
            advantages = np.array(advantages, dtype=np.float32)

            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            # Extract states, actions, and action probabilities
            states = np.array([self._preprocess_state(experience[0]) for experience in memory])
            actions = np.array([experience[1] for experience in memory])
            action_probs = np.array([experience[5] for experience in memory])

            # Create one-hot actions
            actions_one_hot = tf.one_hot(actions, self.action_size)

            # Calculate loss
            with tf.GradientTape() as tape:
                # Forward pass
                policy, values = self.local_model(states)
                values = tf.squeeze(values)

                # Calculate policy loss
                selected_action_probs = tf.reduce_sum(policy * actions_one_hot, axis=1)
                log_probs = tf.math.log(selected_action_probs + 1e-10)
                policy_loss = -tf.reduce_mean(log_probs * advantages)

                # Calculate value loss
                value_loss = tf.reduce_mean(tf.square(returns - values))

                # Calculate entropy
                entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1))

                # Calculate total loss
                total_loss = policy_loss + 0.5 * value_loss - self.entropy_beta * entropy

            # Calculate gradients
            grads = tape.gradient(total_loss, self.local_model.trainable_variables)

            # Apply gradients to global model
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))

            # Update local model with global weights
            self.local_model.set_weights(self.global_model.get_weights())

            return total_loss.numpy()

        except Exception as e:
            self.logger.error(f"Error updating global model: {e}")
            self.logger.error(traceback.format_exc())
            return 0.0
