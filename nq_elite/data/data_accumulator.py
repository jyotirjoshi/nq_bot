#!/usr/bin/env python3
"""
Elite Data Accumulator Module for NQ Alpha Elite

This module provides advanced data storage and management capabilities
for market data, with efficient compression and retrieval.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import traceback
from collections import deque

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.DataAccumulator")

class EliteDataAccumulator:
    """
    Advanced data accumulation system with efficient storage and retrieval.
    Provides high-performance data management for the trading system.
    """
    
    def __init__(self, max_points=10000, logger=None):
        """
        Initialize the data accumulator
        
        Args:
            max_points (int): Maximum data points to store in memory
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.DataAccumulator")
        self.max_points = max_points
        
        # Data storage
        self.data = deque(maxlen=max_points)
        self.data_count = 0
        self.last_update = None
        
        # Performance metrics
        self.metrics = {
            'points_added': 0,
            'points_retrieved': 0,
            'compressions': 0,
            'saves': 0,
            'loads': 0
        }
        
        # Database connection (optional)
        self.db_connection = None
        self.db_table = None
        
        # Create data directory
        self.data_dir = os.path.join(config.DATA_DIR, "accumulated")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info(f"Elite Data Accumulator initialized with {max_points} max points")
    
    def add_data_point(self, data_point):
        """
        Add a data point to the buffer with elite error handling
        
        Args:
            data_point: Dictionary or object with market data
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to dictionary if not already
            if hasattr(data_point, 'to_dict'):
                data_point = data_point.to_dict()
            
            # Get current timestamp if missing
            if 'timestamp' not in data_point:
                data_point['timestamp'] = datetime.now()
            
            # Add to deque
            self.data.append(data_point)
            
            # Update timestamp with proper datetime reference
            self.last_update = datetime.now()
            
            # Update data count
            self.data_count += 1
            self.metrics['points_added'] += 1
            
            # Insert into database if configured
            if self.db_connection and self.db_table:
                self._insert_to_db(data_point)
            
            return True
            
        except Exception as e:
            # Elite error handling with full traceback
            if self.logger:
                self.logger.error(f"Error adding data point: {e}")
                self.logger.error(traceback.format_exc())
            print(f"Critical buffer error: {e}")
            return False
    
    def get_data(self, count=None, as_dataframe=True):
        """
        Get accumulated data
        
        Args:
            count (int, optional): Number of most recent data points to retrieve
            as_dataframe (bool): Return as pandas DataFrame if True, else list
            
        Returns:
            DataFrame or list: Accumulated data
        """
        try:
            # Update metrics
            self.metrics['points_retrieved'] += 1
            
            # Return empty result if no data
            if not self.data:
                return pd.DataFrame() if as_dataframe else []
            
            # Get requested number of points
            if count is None or count >= len(self.data):
                data_subset = list(self.data)
            else:
                data_subset = list(self.data)[-count:]
            
            # Convert to DataFrame if requested
            if as_dataframe:
                return pd.DataFrame(data_subset)
            else:
                return data_subset
                
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame() if as_dataframe else []
    
    def save_data(self, filename=None):
        """
        Save accumulated data to disk
        
        Args:
            filename (str, optional): Custom filename
            
        Returns:
            bool: Success status
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"accumulated_data_{timestamp}.pkl"
            
            # Ensure .pkl extension
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            
            # Create full path
            filepath = os.path.join(self.data_dir, filename)
            
            # Save data
            with open(filepath, 'wb') as f:
                pickle.dump(list(self.data), f)
            
            # Update metrics
            self.metrics['saves'] += 1
            
            self.logger.info(f"Data saved to {filepath} ({len(self.data)} points)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def load_data(self, filename):
        """
        Load data from disk
        
        Args:
            filename (str): Filename to load
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure .pkl extension
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            
            # Create full path
            filepath = os.path.join(self.data_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.error(f"File not found: {filepath}")
                return False
            
            # Load data
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Clear current data
            self.data.clear()
            
            # Add loaded data
            for point in loaded_data:
                self.data.append(point)
            
            # Update metrics
            self.data_count = len(self.data)
            self.metrics['loads'] += 1
            self.last_update = datetime.now()
            
            self.logger.info(f"Data loaded from {filepath} ({len(self.data)} points)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def compress_data(self, compression_ratio=0.5):
        """
        Compress data by sampling to reduce memory usage
        
        Args:
            compression_ratio (float): Ratio of data to keep (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if not enough data
            if len(self.data) < 100:
                return False
            
            # Save original data count
            original_count = len(self.data)
            
            # Calculate number of points to keep
            keep_count = max(100, int(original_count * compression_ratio))
            
            # Keep all recent data (30% of keep_count)
            recent_count = max(10, int(keep_count * 0.3))
            recent_data = list(self.data)[-recent_count:]
            
            # Sample from older data
            older_data = list(self.data)[:-recent_count]
            sample_count = keep_count - recent_count
            
            if sample_count > 0 and older_data:
                # Use systematic sampling for older data
                step = len(older_data) / sample_count
                indices = [int(i * step) for i in range(sample_count)]
                sampled_older_data = [older_data[i] for i in indices if i < len(older_data)]
                
                # Combine sampled older data with recent data
                compressed_data = sampled_older_data + recent_data
            else:
                compressed_data = recent_data
            
            # Replace data with compressed version
            self.data.clear()
            for point in compressed_data:
                self.data.append(point)
            
            # Update metrics
            self.metrics['compressions'] += 1
            
            self.logger.info(f"Data compressed: {original_count} → {len(self.data)} points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_statistics(self):
        """
        Get statistics about the accumulated data
        
        Returns:
            dict: Statistics
        """
        try:
            if not self.data:
                return {
                    'count': 0,
                    'memory_usage': 0,
                    'first_timestamp': None,
                    'last_timestamp': None,
                    'duration': None
                }
            
            # Convert to DataFrame for statistics
            df = pd.DataFrame(self.data)
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Get timestamps
            if 'timestamp' in df.columns:
                first_timestamp = df['timestamp'].min()
                last_timestamp = df['timestamp'].max()
                
                # Calculate duration
                if isinstance(first_timestamp, (datetime, np.datetime64)):
                    duration = last_timestamp - first_timestamp
                    duration_seconds = duration.total_seconds()
                else:
                    duration_seconds = None
            else:
                first_timestamp = None
                last_timestamp = None
                duration_seconds = None
            
            # Return statistics
            return {
                'count': len(self.data),
                'memory_usage': memory_usage,
                'first_timestamp': first_timestamp,
                'last_timestamp': last_timestamp,
                'duration': duration_seconds,
                'metrics': self.metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {'count': len(self.data), 'error': str(e)}
    
    def _insert_to_db(self, data_point):
        """
        Insert data point to database (if configured)
        
        Args:
            data_point: Data point to insert
            
        Returns:
            bool: Success status
        """
        # This is a placeholder for database integration
        # Implement specific database logic here if needed
        return True
