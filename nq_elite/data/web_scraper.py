#!/usr/bin/env python3
"""
Web Scraper Module for NQ Alpha Elite

This module provides specialized web scraping capabilities for collecting
real-time market data for NASDAQ 100 E-mini futures.
"""
import os
import sys
import logging
import requests
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import re
import json
import traceback
from collections import deque
import threading

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config

# Configure logging
logger = logging.getLogger("NQAlpha.WebScraper")

class NQDirectFeed:
    """
    Direct web scraping feed for NQ futures data
    
    This class provides specialized web scraping capabilities for collecting
    real-time market data for NASDAQ 100 E-mini futures without using APIs.
    """
    
    def __init__(self, clean_start=True, logger=None):
        """
        Initialize the web scraper
        
        Args:
            clean_start (bool): Whether to clear existing data
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.WebScraper")
        
        # Initialize data storage
        self.market_data = []
        self.max_data_points = config.MARKET_DATA_CONFIG['max_data_points']
        
        # Initialize state
        self.running = False
        self.thread = None
        self.last_price = None
        self.bid = None
        self.ask = None
        self.spread = None
        self.volume = 0
        self.tick_count = 0
        self.last_update_time = None
        self.data_source = None
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update(config.MARKET_DATA_CONFIG['headers'])
        
        # Initialize sources
        self.sources = [
            {
                'name': 'cme',
                'url': 'https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.quotes.html',
                'method': self._scrape_cme
            },
            {
                'name': 'tradingview',
                'url': 'https://www.tradingview.com/symbols/CME_MINI-NQ1!/',
                'method': self._scrape_tradingview
            },
            {
                'name': 'investing',
                'url': 'https://www.investing.com/indices/nasdaq-100-futures',
                'method': self._scrape_investing
            },
            {
                'name': 'barchart',
                'url': 'https://www.barchart.com/futures/quotes/NQ*0',
                'method': self._scrape_barchart
            }
        ]
        
        # Initialize metrics
        self.metrics = {
            'requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_points': 0,
            'start_time': datetime.now(),
            'last_update': None
        }
        
        # Data accumulator reference (will be set externally)
        self.data_accumulator = None
        
        # Clear existing data if clean start
        if clean_start:
            self.market_data = []
        
        self.logger.info("NQ Direct Feed initialized")
    
    def start(self, interval=2.0):
        """
        Start the web scraper
        
        Args:
            interval (float): Update interval in seconds
        """
        if self.running:
            self.logger.warning("Web scraper already running")
            return
        
        self.logger.info(f"Starting web scraper with {interval}s update interval")
        
        try:
            # Set running flag
            self.running = True
            self.metrics['start_time'] = datetime.now()
            
            # Start in background thread
            self.thread = threading.Thread(
                target=self._scraper_thread,
                args=(interval,),
                name="WebScraperThread"
            )
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Web scraper thread started")
            
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting web scraper: {e}")
    
    def stop(self):
        """Stop the web scraper"""
        if not self.running:
            self.logger.warning("Web scraper not running")
            return
        
        self.logger.info("Stopping web scraper")
        
        try:
            # Set running flag
            self.running = False
            
            # Wait for thread to complete
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)
            
            # Save data
            if self.market_data:
                self._save_market_data()
            
            self.logger.info("Web scraper stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping web scraper: {e}")
    
    def _scraper_thread(self, interval):
        """
        Background thread for web scraping
        
        Args:
            interval (float): Update interval in seconds
        """
        self.logger.info("Web scraper thread running")
        
        try:
            while self.running:
                try:
                    start_time = time.time()
                    
                    # Scrape market data
                    self.update_data()
                    
                    # Sleep for remaining interval time
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, interval - elapsed)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Error in web scraper loop: {e}")
                    time.sleep(1.0)
            
        except Exception as e:
            self.logger.error(f"Fatal error in web scraper thread: {e}")
        
        self.logger.info("Web scraper thread stopped")
    
    def update_data(self):
        """Update market data from web sources"""
        try:
            # Try each source until successful
            for source in self.sources:
                try:
                    # Scrape data from source
                    data = source['method']()
                    
                    if data:
                        # Update metrics
                        self.metrics['successful_requests'] += 1
                        self.metrics['data_points'] += 1
                        self.metrics['last_update'] = datetime.now()
                        
                        # Update data
                        self._update_market_data(data, source['name'])
                        
                        return True
                    
                except Exception as e:
                    self.logger.debug(f"Error scraping {source['name']}: {e}")
                    self.metrics['failed_requests'] += 1
            
            self.logger.warning("Failed to scrape data from any source")
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False
    
    def _update_market_data(self, data, source):
        """
        Update market data with new values
        
        Args:
            data (dict): Market data
            source (str): Data source
        """
        try:
            # Get current time
            current_time = datetime.now()
            
            # Extract data
            price = data.get('price')
            bid = data.get('bid')
            ask = data.get('ask')
            volume = data.get('volume', 0)
            
            if price is None:
                self.logger.warning(f"Skipping update: Price is None from source {source}")
                return
            
            # Calculate price change
            price_change = 0.0
            if self.last_price:
                price_change = price - self.last_price
            
            # Update state
            self.last_price = price
            self.bid = bid
            self.ask = ask
            self.spread = ask - bid if bid is not None and ask is not None else None
            self.volume += volume
            self.tick_count += 1
            self.last_update_time = current_time
            self.data_source = source
            
            # Create market data point
            market_data_point = {
                'timestamp': current_time,
                'price': price,
                'bid': bid,
                'ask': ask,
                'spread': self.spread,
                'volume': volume,
                'source': source
            }
            
            # Add to market data
            self.market_data.append(market_data_point)
            
            # Limit market data size
            if len(self.market_data) > self.max_data_points:
                self.market_data = self.market_data[-self.max_data_points:]
            
            # Add to data accumulator if available
            if hasattr(self, 'data_accumulator') and self.data_accumulator is not None:
                try:
                    self.data_accumulator.add_data_point(market_data_point)
                except Exception as e:
                    self.logger.error(f"Error adding to data accumulator: {e}")
            
            self.logger.debug(f"Updated market data: {price} from {source}")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _save_market_data(self):
        """Save market data to disk"""
        try:
            if not self.market_data:
                return
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"NQ_{timestamp}.csv"
            filepath = os.path.join(config.DATA_DIR, "market_data", filename)
            
            # Convert to DataFrame
            df = pd.DataFrame(self.market_data)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Saved market data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
    
    def get_market_data(self, count=100, lookback=None):
        """
        Get historical market data
        
        Args:
            count (int): Number of data points
            lookback (int, optional): Alias for count (for compatibility)
            
        Returns:
            DataFrame: Market data
        """
        try:
            if not self.market_data:
                return pd.DataFrame()
            
            # Use lookback if provided (for compatibility with other systems)
            num_records = lookback if lookback is not None else count
            
            # Return recent data
            recent_data = self.market_data[-num_records:]
            
            # Convert to DataFrame
            return pd.DataFrame(recent_data)
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
