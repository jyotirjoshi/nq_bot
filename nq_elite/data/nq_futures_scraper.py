# Modified nq_elite/data/nq_futures_scraper.py
import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import os
import websocket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import pickle
from queue import Queue

class NQFuturesScraper:
    """Advanced web scraper for Nasdaq 100 E-mini futures data with high-frequency capability"""
    
    def __init__(self, config=None):
        """Initialize the scraper
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("NQAlpha.Scraper")
        
        # Default configuration
        self.config = {
            'cache_dir': 'cache/nq_futures',
            'data_dir': 'data/nq_futures',
            'use_cache': True,
            'update_frequency': {
                '10s': 10,     # New high-frequency intervals
                '30s': 30,     # New high-frequency intervals
                '1m': 60,
                '5m': 300,
                '15m': 900, 
                '1h': 3600,
                '4h': 14400,
                'daily': 86400
            },
            'api_endpoints': {
                'primary': 'https://futures-api.example.com/v2/',
                'secondary': 'https://alt-futures-data.example.com/api/',
                'websocket': 'wss://futures-stream.example.com/ws'
            },
            'api_keys': {
                'primary': '',
                'secondary': ''
            },
            'timeframes': ['10s', '30s', '1m', '5m', '15m', '1h', '4h', 'daily'],
            'max_retries': 3,
            'retry_delay': 5,
            'concurrency': 5,
            'verify_data_integrity': True,
            'data_validation': {
                'price_sanity_check': True,
                'volume_validation': True,
                'duplicate_detection': True,
                'gap_detection': True,
                'microstructure_validation': True  # New for high-frequency data
            },
            'collection_hours': {
                'regular_session': True,
                'extended_hours': True,
                'weekend_data': False
            },
            'chrome_driver_path': 'drivers/chromedriver',
            'headless': True,
            'news_sources': ['bloomberg', 'reuters', 'cnbc', 'wsj'],
            'economic_calendar_sources': ['forexfactory', 'investing'],
            'proxy': None,
            'websocket_enabled': True,  # Enable for real-time data
            'hf_buffer_size': 1000,     # Store 1000 high-frequency bars in memory
            'adaptive_sampling': True   # Adaptive time sampling for market regimes
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
        
        # Create data and cache directories
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Initialize state
        self.running = False
        self.stop_event = threading.Event()
        self.threads = []
        self.websocket = None
        self.data_queue = Queue()
        self.data_buffer = {tf: pd.DataFrame() for tf in self.config['timeframes']}
        self.driver = None
        self.last_update = {tf: datetime.now() - timedelta(days=1) for tf in self.config['timeframes']}
        self.microstructure_data = []  # For 10s data
        
        # Initialize data locks
        self.data_locks = {tf: threading.Lock() for tf in self.config['timeframes']}
        
        # Signal new data available
        self.new_data_event = threading.Event()
        self.hf_data_event = threading.Event()  # Specific for high-frequency data
        
        self.logger.info("NQ Futures Scraper initialized")
    
    def _update_config(self, target, source):
        """Update configuration recursively
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def start(self):
        """Start the scraper
        
        Returns:
            bool: Success flag
        """
        try:
            if self.running:
                self.logger.warning("Scraper already running")
                return False
            
            self.logger.info("Starting NQ Futures Scraper")
            
            # Set running flag and clear stop event
            self.running = True
            self.stop_event.clear()
            self.threads.append(threading.Thread(
                target=self.collect_high_frequency_data,
                name="HighFrequencyDataThread",
                daemon=True
            ))
            # Initialize web driver if needed
            if self.config['headless']:
                self._init_web_driver()
            
            # Start data collection threads
            self._start_collection_threads()
            
            # Start websocket connection for real-time data if enabled
            if self.config['websocket_enabled']:
                self._start_websocket_connection()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting scraper: {str(e)}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the scraper
        
        Returns:
            bool: Success flag
        """
        try:
            if not self.running:
                self.logger.warning("Scraper not running")
                return False
            
            self.logger.info("Stopping NQ Futures Scraper")
            
            # Set stop event and clear running flag
            self.stop_event.set()
            self.running = False
            
            # Wait for all threads to exit
            for thread in self.threads:
                thread.join(timeout=5)
            
            # Close websocket if open
            if self.websocket:
                self.websocket.close()
                self.websocket = None
            
            # Close web driver if initialized
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping scraper: {str(e)}")
            self.running = False
            return False
    
    def _init_web_driver(self):
        """Initialize web driver for scraping"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            self.driver = webdriver.Chrome(executable_path=self.config['chrome_driver_path'], options=chrome_options)
            self.logger.debug("Web driver initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing web driver: {str(e)}")
            self.driver = None
    
    def _start_collection_threads(self):
        """Start data collection threads"""
        try:
            # Start price data collection threads for each timeframe
            for timeframe in self.config['timeframes']:
                thread = threading.Thread(
                    target=self._collect_price_data,
                    args=(timeframe,),
                    name=f"PriceDataThread-{timeframe}",
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
            
            # Special thread for 10-second data (requires different handling)
            if '10s' in self.config['timeframes']:
                thread = threading.Thread(
                    target=self._collect_high_frequency_data,
                    name="HighFrequencyDataThread",
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
            
            # Start news data collection thread
            thread = threading.Thread(
                target=self._collect_news_data,
                name="NewsDataThread",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
            # Start economic calendar data collection thread
            thread = threading.Thread(
                target=self._collect_economic_data,
                name="EconomicDataThread",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
            # Start data processing thread
            thread = threading.Thread(
                target=self._process_data_queue,
                name="DataProcessingThread",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
            self.logger.info("Data collection threads started")
            
        except Exception as e:
            self.logger.error(f"Error starting collection threads: {str(e)}")
    
    def _start_websocket_connection(self):
        """Start websocket connection for real-time data"""
        try:
            if not self.config['api_keys']['primary']:
                self.logger.warning("No API key provided, websocket connection disabled")
                return
            
            # Start websocket thread
            thread = threading.Thread(
                target=self._websocket_thread,
                name="WebsocketThread",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
            self.logger.info("Websocket connection started")
            
        except Exception as e:
            self.logger.error(f"Error starting websocket connection: {str(e)}")
    
    def _websocket_thread(self):
        """Websocket thread for real-time data"""
        try:
            # WebSocket URL with API key
            ws_url = f"{self.config['api_endpoints']['websocket']}?token={self.config['api_keys']['primary']}"
            
            # WebSocket connection
            self.websocket = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close,
                on_open=self._on_websocket_open
            )
            
            # Run WebSocket connection
            while self.running and not self.stop_event.is_set():
                self.websocket.run_forever()
                
                # Reconnect if disconnected and still running
                if self.running and not self.stop_event.is_set():
                    self.logger.info("Websocket disconnected, reconnecting...")
                    time.sleep(5)
            
        except Exception as e:
            self.logger.error(f"Error in websocket thread: {str(e)}")
    
    def _on_websocket_message(self, ws, message):
        """Handle websocket message
        
        Args:
            ws: Websocket connection
            message: Message received
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Add to data queue for processing
            self.data_queue.put({'type': 'websocket', 'data': data})
            
            # Signal new data available for high-frequency
            self.hf_data_event.set()
            
        except Exception as e:
            self.logger.error(f"Error processing websocket message: {str(e)}")
    
    def _on_websocket_error(self, ws, error):
        """Handle websocket error
        
        Args:
            ws: Websocket connection
            error: Error received
        """
        self.logger.error(f"Websocket error: {str(error)}")
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle websocket close
        
        Args:
            ws: Websocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        self.logger.info("Websocket connection closed")
    
    def _on_websocket_open(self, ws):
        """Handle websocket open
        
        Args:
            ws: Websocket connection
        """
        try:
            # Subscribe to NQ futures
            subscribe_msg = {
                'type': 'subscribe',
                'symbol': 'NQ',
                'interval': '10s'  # Subscribe to 10-second data
            }
            
            ws.send(json.dumps(subscribe_msg))
            self.logger.info("Subscribed to NQ futures data")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to websocket: {str(e)}")
    
    def _collect_high_frequency_data(self):
        """Collect high-frequency data (10-second intervals)"""
        try:
            while self.running and not self.stop_event.is_set():
                # Check if trading hours
                if not self._is_trading_hours():
                    time.sleep(60)  # Sleep for 1 minute and check again
                    continue
                
                try:
                    # Get high-frequency data
                    url = f"{self.config['api_endpoints']['primary']}quotes/NQ/microbar"
                    headers = {'Authorization': f"Bearer {self.config['api_keys']['primary']}"}
                    
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Validate data
                        if self._validate_high_frequency_data(data):
                            # Process and store data
                            self._process_high_frequency_data(data)
                            
                            # Update last update time
                            self.last_update['10s'] = datetime.now()
                            
                            # Signal new data available
                            self.hf_data_event.set()
                    else:
                        self.logger.warning(f"Failed to get high-frequency data: {response.status_code}")
                
                except Exception as e:
                    self.logger.error(f"Error collecting high-frequency data: {str(e)}")
                
                # Sleep for remaining time until next interval
                sleep_time = max(1, self.config['update_frequency']['10s'] - (datetime.now() - self.last_update['10s']).total_seconds())
                time.sleep(sleep_time)
            
        except Exception as e:
            self.logger.error(f"Error in high-frequency data collection thread: {str(e)}")
    
    def _validate_high_frequency_data(self, data):
        """Validate high-frequency data
        
        Args:
            data: High-frequency data
            
        Returns:
            bool: Validation result
        """
        try:
            # Check if data is empty
            if not data or 'bars' not in data or not data['bars']:
                return False
            
            # Check if data has required fields
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for bar in data['bars']:
                if not all(field in bar for field in required_fields):
                    return False
            
            # Validate price ranges (if enabled)
            if self.config['data_validation']['price_sanity_check']:
                for bar in data['bars']:
                    # Check OHLC relationships
                    if not (bar['low'] <= bar['open'] <= bar['high'] and 
                            bar['low'] <= bar['close'] <= bar['high']):
                        return False
            
            # Validate volume (if enabled)
            if self.config['data_validation']['volume_validation']:
                for bar in data['bars']:
                    if bar['volume'] < 0:
                        return False
            
            # Microstructure validation (special for high-frequency)
            if self.config['data_validation']['microstructure_validation']:
                # Check for unrealistic price jumps between consecutive bars
                if len(data['bars']) > 1:
                    for i in range(1, len(data['bars'])):
                        prev_close = data['bars'][i-1]['close']
                        curr_open = data['bars'][i]['open']
                        
                        # Detect unrealistic gaps (more than 0.5% in 10 seconds)
                        price_change_pct = abs(curr_open - prev_close) / prev_close
                        if price_change_pct > 0.005:
                            self.logger.warning(f"Suspicious price gap detected: {price_change_pct:.2%}")
                            # Don't reject, but log warning
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating high-frequency data: {str(e)}")
            return False
    
    def _process_high_frequency_data(self, data):
        """Process high-frequency data
        
        Args:
            data: High-frequency data
        """
        try:
            # Convert data to DataFrame
            bars = data['bars']
            df = pd.DataFrame(bars)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add calculated fields for microstructure analysis
            df['price_velocity'] = df['close'].diff() / df['timestamp'].diff().dt.total_seconds()
            df['volume_imbalance'] = df['volume'].diff() / df['volume'].rolling(window=3).mean()
            
            # Store in memory buffer (circular buffer)
            with self.data_locks['10s']:
                # Append to existing data
                if self.data_buffer['10s'].empty:
                    self.data_buffer['10s'] = df
                else:
                    self.data_buffer['10s'] = pd.concat([self.data_buffer['10s'], df])
                
                # Keep only last N bars
                max_bars = self.config['hf_buffer_size']
                if len(self.data_buffer['10s']) > max_bars:
                    self.data_buffer['10s'] = self.data_buffer['10s'].iloc[-max_bars:]
            
            # Also store for microstructure analysis
            self.microstructure_data.extend(bars)
            if len(self.microstructure_data) > max_bars * 3:
                self.microstructure_data = self.microstructure_data[-max_bars * 3:]
            
            # Save to disk periodically (every 100 bars)
            if len(df) % 100 == 0:
                self._save_high_frequency_data()
            
        except Exception as e:
            self.logger.error(f"Error processing high-frequency data: {str(e)}")
    def collect_high_frequency_data(self):
        """Collect high-frequency data (10-second and 1-minute intervals)"""
        try:
            while self.running and not self.stop_event.is_set():
                # Check if trading hours
                if not self._is_trading_hours():
                    time.sleep(60)  # Sleep for 1 minute and check again
                    continue
                
                try:
                    # Get current time
                    now = datetime.now()
                    
                    # For 10-second data
                    if now.second % 10 == 0:
                        # Get high-frequency data from API or websocket
                        hf_data = self._fetch_hf_data('10s')
                        
                        if hf_data is not None:
                            # Process and store data
                            self._process_hf_data(hf_data, '10s')
                            
                            # Signal new data available
                            self.hf_data_event.set()
                    
                    # For 1-minute data
                    if now.second == 0:
                        # Get 1-minute data
                        minute_data = self._fetch_hf_data('1m')
                        
                        if minute_data is not None:
                            # Process and store data
                            self._process_hf_data(minute_data, '1m')
                            
                            # Signal new data available
                            self.new_data_event.set()
                    
                    # Sleep for approximately 1 second, but wake precisely at next 10-second mark
                    next_10s = (now.replace(microsecond=0) + timedelta(seconds=10 - now.second % 10))
                    sleep_time = (next_10s - datetime.now()).total_seconds()
                    sleep_time = max(0.1, min(1.0, sleep_time))  # Ensure between 0.1 and 1.0 seconds
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting high-frequency data: {str(e)}")
                    time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error in high-frequency data collection thread: {str(e)}")
            
    def _fetch_hf_data(self, timeframe):
        """Fetch high-frequency data
        
        Args:
            timeframe: Data timeframe ('10s' or '1m')
            
        Returns:
            dict: High-frequency data or None on failure
        """
        try:
            # Determine API endpoint based on timeframe
            if timeframe == '10s':
                endpoint = f"{self.config['api_endpoints']['primary']}quotes/NQ/microbar"
            else:  # 1m
                endpoint = f"{self.config['api_endpoints']['primary']}quotes/NQ/1m"
            
            # Add API key if available
            headers = {}
            if self.config['api_keys']['primary']:
                headers['Authorization'] = f"Bearer {self.config['api_keys']['primary']}"
            
            # Make API request with timeout
            response = requests.get(endpoint, headers=headers, timeout=2.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to fetch {timeframe} data: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout fetching {timeframe} data")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching {timeframe} data: {str(e)}")
            return None
            
    def _process_hf_data(self, data, timeframe):
        """Process high-frequency data
        
        Args:
            data: High-frequency data
            timeframe: Data timeframe ('10s' or '1m')
        """
        try:
            # Extract bars from response
            if 'bars' not in data or not data['bars']:
                return
                
            bars = data['bars']
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add calculated fields for microstructure analysis
            if timeframe == '10s':
                # Add price velocity (rate of price change)
                df['price_velocity'] = df['close'].diff() / df['timestamp'].diff().dt.total_seconds()
                
                # Add volume imbalance (relative to recent average)
                if 'volume' in df.columns:
                    df['volume_imbalance'] = df['volume'] / df['volume'].rolling(window=6).mean()
                
                # Add microstructure noise estimation
                df['price_noise'] = (df['high'] - df['low']) / df['close']
            
            # Store in memory buffer
            with self.data_locks[timeframe]:
                # Append to existing data
                if self.data_buffer[timeframe].empty:
                    self.data_buffer[timeframe] = df
                else:
                    self.data_buffer[timeframe] = pd.concat([self.data_buffer[timeframe], df])
                
                # Keep only last N bars
                max_bars = self.config.get('hf_buffer_size', 1000)
                if len(self.data_buffer[timeframe]) > max_bars:
                    self.data_buffer[timeframe] = self.data_buffer[timeframe].iloc[-max_bars:]
            
            # Save to disk
            self._save_hf_data(df, timeframe)
            
            # Log data receipt
            self.logger.debug(f"Processed {len(df)} {timeframe} bars for NQ futures")
            
        except Exception as e:
            self.logger.error(f"Error processing {timeframe} data: {str(e)}")
            
    def _save_hf_data(self, df, timeframe):
        """Save high-frequency data to disk
        
        Args:
            df: Data DataFrame
            timeframe: Data timeframe
        """
        try:
            if df.empty:
                return
            
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"nq_futures_{timeframe}_{date_str}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Check if file exists
            file_exists = os.path.exists(filepath)
            
            # Append to file or create new file
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            
            # Save to file
            df.to_csv(filepath, mode=mode, header=header, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving {timeframe} data: {str(e)}")
    def _save_high_frequency_data(self):
        """Save high-frequency data to disk"""
        try:
            # Get data from buffer
            with self.data_locks['10s']:
                df = self.data_buffer['10s'].copy()
            
            if df.empty:
                return
            
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"nq_futures_10s_{date_str}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Save to file (append mode)
            if os.path.exists(filepath):
                # Read existing file to check for duplicates
                existing_df = pd.read_csv(filepath)
                
                # Convert timestamp to datetime
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Filter out duplicates
                new_data = df[~df['timestamp'].isin(existing_df['timestamp'])]
                
                if not new_data.empty:
                    # Append new data to file
                    new_data.to_csv(filepath, mode='a', header=False, index=False)
            else:
                # Create new file
                df.to_csv(filepath, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving high-frequency data: {str(e)}")
    
    def _collect_price_data(self, timeframe):
        """Collect price data for a specific timeframe
        
        Args:
            timeframe: Data timeframe
        """
        try:
            # Skip high-frequency data, handled separately
            if timeframe == '10s' or timeframe == '30s':
                return
                
            while self.running and not self.stop_event.is_set():
                # Check update interval
                time_since_update = (datetime.now() - self.last_update[timeframe]).total_seconds()
                
                if time_since_update >= self.config['update_frequency'][timeframe]:
                    try:
                        # Get latest data
                        latest_data = self._fetch_latest_price_data(timeframe)
                        
                        if latest_data is not None:
                            # Process and store data
                            self._process_price_data(latest_data, timeframe)
                            
                            # Update last update time
                            self.last_update[timeframe] = datetime.now()
                            
                            # Signal new data available
                            self.new_data_event.set()
                    
                    except Exception as e:
                        self.logger.error(f"Error fetching price data for {timeframe}: {str(e)}")
                
                # Sleep for remaining time until next update
                sleep_time = max(1, self.config['update_frequency'][timeframe] - (datetime.now() - self.last_update[timeframe]).total_seconds())
                time.sleep(sleep_time)
            
        except Exception as e:
            self.logger.error(f"Error in price data collection thread for {timeframe}: {str(e)}")
    
    def _fetch_latest_price_data(self, timeframe):
        """Fetch latest price data for a specific timeframe
        
        Args:
            timeframe: Data timeframe
            
        Returns:
            dict: Latest price data or None on failure
        """
        try:
            # Construct API URL
            url = f"{self.config['api_endpoints']['primary']}quote/NQ/{timeframe}"
            
            # Add API key if available
            headers = {}
            if self.config['api_keys']['primary']:
                headers['Authorization'] = f"Bearer {self.config['api_keys']['primary']}"
            
            # Make API request
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to fetch price data for {timeframe}: {response.status_code}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error fetching price data for {timeframe}: {str(e)}")
            return None
    
    def _process_price_data(self, data, timeframe):
        """Process price data
        
        Args:
            data: Price data
            timeframe: Data timeframe
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data['bars'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add date column
            df['date'] = df['timestamp'].dt.date
            
            # Store in memory buffer
            with self.data_locks[timeframe]:
                # Replace existing data
                self.data_buffer[timeframe] = df
            
            # Save to disk
            self._save_price_data(df, timeframe)
            
        except Exception as e:
            self.logger.error(f"Error processing price data for {timeframe}: {str(e)}")
    
    def _save_price_data(self, df, timeframe):
        """Save price data to disk
        
        Args:
            df: Price data DataFrame
            timeframe: Data timeframe
        """
        try:
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"nq_futures_{timeframe}_{date_str}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Save to file
            df.to_csv(filepath, index=False)
            
            # Also save latest data
            latest_filepath = os.path.join(self.config['data_dir'], f"nq_futures_{timeframe}_latest.csv")
            df.to_csv(latest_filepath, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving price data for {timeframe}: {str(e)}")
    
    def _collect_news_data(self):
        """Collect news data"""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    # Collect news from configured sources
                    for source in self.config['news_sources']:
                        news_data = self._fetch_news_data(source)
                        
                        if news_data is not None:
                            # Process and store news data
                            self._process_news_data(news_data, source)
                
                except Exception as e:
                    self.logger.error(f"Error collecting news data: {str(e)}")
                
                # Sleep for 15 minutes before next update (news doesn't need high frequency)
                time.sleep(900)
            
        except Exception as e:
            self.logger.error(f"Error in news data collection thread: {str(e)}")
    
    def _fetch_news_data(self, source):
        """Fetch news data from a specific source
        
        Args:
            source: News source
            
        Returns:
            dict: News data or None on failure
        """
        try:
            # Use web scraping for news
            if self.driver is None:
                self.logger.warning("Web driver not initialized for news scraping")
                return None
            
            # URLs for different sources
            urls = {
                'bloomberg': 'https://www.bloomberg.com/markets',
                'reuters': 'https://www.reuters.com/markets',
                'cnbc': 'https://www.cnbc.com/markets',
                'wsj': 'https://www.wsj.com/news/markets'
            }
            
            if source not in urls:
                self.logger.warning(f"Unknown news source: {source}")
                return None
            
            # Navigate to URL
            self.driver.get(urls[source])
            
            # Wait for page to load
            time.sleep(5)
            
            # Get page content
            page_content = self.driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract news headlines and summaries (site-specific selectors)
            news_items = []
            
            if source == 'bloomberg':
                articles = soup.select('article')
                for article in articles[:10]:  # Get top 10 articles
                    headline_elem = article.select_one('h3')
                    summary_elem = article.select_one('p')
                    
                    if headline_elem:
                        headline = headline_elem.text.strip()
                        summary = summary_elem.text.strip() if summary_elem else ""
                        
                        news_items.append({
                            'headline': headline,
                            'summary': summary,
                            'source': source,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Similar parsing for other sources...
            
            return {'news': news_items}
            
        except Exception as e:
            self.logger.error(f"Error fetching news data from {source}: {str(e)}")
            return None
    
    def _process_news_data(self, data, source):
        """Process news data
        
        Args:
            data: News data
            source: News source
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data['news'])
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().isoformat()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save to disk
            self._save_news_data(df, source)
            
        except Exception as e:
            self.logger.error(f"Error processing news data from {source}: {str(e)}")
    
    def _save_news_data(self, df, source):
        """Save news data to disk
        
        Args:
            df: News data DataFrame
            source: News source
        """
        try:
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"nq_news_{source}_{date_str}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Save to file (append mode)
            if os.path.exists(filepath):
                # Read existing file to check for duplicates
                existing_df = pd.read_csv(filepath)
                
                # Convert timestamp to datetime
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Filter out duplicates (based on headline)
                new_data = df[~df['headline'].isin(existing_df['headline'])]
                
                if not new_data.empty:
                    # Append new data to file
                    new_data.to_csv(filepath, mode='a', header=False, index=False)
            else:
                # Create new file
                df.to_csv(filepath, index=False)
            
            # Also save latest news
            latest_filepath = os.path.join(self.config['data_dir'], f"nq_news_{source}_latest.csv")
            df.to_csv(latest_filepath, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving news data from {source}: {str(e)}")
    
    def _collect_economic_data(self):
        """Collect economic calendar data"""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    # Collect economic data from configured sources
                    for source in self.config['economic_calendar_sources']:
                        economic_data = self._fetch_economic_data(source)
                        
                        if economic_data is not None:
                            # Process and store economic data
                            self._process_economic_data(economic_data, source)
                
                except Exception as e:
                    self.logger.error(f"Error collecting economic data: {str(e)}")
                
                # Sleep for 1 hour before next update (economic data doesn't need high frequency)
                time.sleep(3600)
            
        except Exception as e:
            self.logger.error(f"Error in economic data collection thread: {str(e)}")
    
    def _fetch_economic_data(self, source):
        """Fetch economic calendar data from a specific source
        
        Args:
            source: Economic calendar source
            
        Returns:
            dict: Economic calendar data or None on failure
        """
        try:
            # Use web scraping for economic calendar
            if self.driver is None:
                self.logger.warning("Web driver not initialized for economic data scraping")
                return None
            
            # URLs for different sources
            urls = {
                'forexfactory': 'https://www.forexfactory.com/calendar',
                'investing': 'https://www.investing.com/economic-calendar/'
            }
            
            if source not in urls:
                self.logger.warning(f"Unknown economic calendar source: {source}")
                return None
            
            # Navigate to URL
            self.driver.get(urls[source])
            
            # Wait for page to load
            time.sleep(5)
            
            # Get page content
            page_content = self.driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract economic events (site-specific selectors)
            events = []
            
            if source == 'forexfactory':
                rows = soup.select('tr.calendar_row')
                for row in rows:
                    date_elem = row.select_one('td.calendar__date')
                    time_elem = row.select_one('td.calendar__time')
                    currency_elem = row.select_one('td.calendar__currency')
                    event_elem = row.select_one('td.calendar__event')
                    impact_elem = row.select_one('td.calendar__impact')
                    
                    if date_elem and event_elem:
                        date_text = date_elem.text.strip()
                        time_text = time_elem.text.strip() if time_elem else ""
                        currency = currency_elem.text.strip() if currency_elem else ""
                        event = event_elem.text.strip()
                        impact = impact_elem.text.strip() if impact_elem else ""
                        
                        events.append({
                            'date': date_text,
                            'time': time_text,
                            'currency': currency,
                            'event': event,
                            'impact': impact,
                            'source': source,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Similar parsing for other sources...
            
            return {'events': events}
            
        except Exception as e:
            self.logger.error(f"Error fetching economic data from {source}: {str(e)}")
            return None
    
    def _process_economic_data(self, data, source):
        """Process economic calendar data
        
        Args:
            data: Economic calendar data
            source: Economic calendar source
        """
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data['events'])
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().isoformat()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save to disk
            self._save_economic_data(df, source)
            
        except Exception as e:
            self.logger.error(f"Error processing economic data from {source}: {str(e)}")
    
    def _save_economic_data(self, df, source):
        """Save economic calendar data to disk
        
        Args:
            df: Economic calendar data DataFrame
            source: Economic calendar source
        """
        try:
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"nq_economic_{source}_{date_str}.csv"
            filepath = os.path.join(self.config['data_dir'], filename)
            
            # Save to file
            df.to_csv(filepath, index=False)
            
            # Also save latest economic data
            latest_filepath = os.path.join(self.config['data_dir'], f"nq_economic_{source}_latest.csv")
            df.to_csv(latest_filepath, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving economic data from {source}: {str(e)}")
    
    def _process_data_queue(self):
        """Process data from queue"""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    # Get data from queue (with timeout)
                    data = self.data_queue.get(timeout=1)
                    
                    # Process data based on type
                    if data['type'] == 'websocket':
                        self._process_websocket_data(data['data'])
                    
                    # Mark task as done
                    self.data_queue.task_done()
                    
                except queue.Empty:
                    # Queue is empty, just continue
                    pass
                
                except Exception as e:
                    self.logger.error(f"Error processing data from queue: {str(e)}")
                
                # Short sleep to prevent CPU thrashing
                time.sleep(0.01)
            
        except Exception as e:
            self.logger.error(f"Error in data processing thread: {str(e)}")
    
    def _process_websocket_data(self, data):
        """Process websocket data
        
        Args:
            data: Websocket data
        """
        try:
            # Check if data is a tick or bar
            if 'type' in data:
                if data['type'] == 'tick':
                    # Process tick data for 10-second bars
                    self._process_tick_data(data)
                elif data['type'] == 'bar':
                    # Process bar data for other timeframes
                    self._process_bar_data(data)
            
        except Exception as e:
            self.logger.error(f"Error processing websocket data: {str(e)}")
    
    def _process_tick_data(self, data):
        """Process tick data for 10-second bars
        
        Args:
            data: Tick data
        """
        try:
            # Extract tick data
            tick = {
                'timestamp': pd.to_datetime(data['timestamp'], unit='ms'),
                'price': data['price'],
                'volume': data['volume'],
                'side': data.get('side', 'unknown')
            }
            
            # Add to microstructure data
            self.microstructure_data.append(tick)
            
            # Limit buffer size
            if len(self.microstructure_data) > self.config['hf_buffer_size'] * 10:
                self.microstructure_data = self.microstructure_data[-self.config['hf_buffer_size'] * 10:]
            
            # Check if we need to update 10-second bar
            now = datetime.now()
            seconds = now.second
            
            # Update 10-second bar every 10 seconds
            if seconds % 10 == 0 and len(self.microstructure_data) > 0:
                # Construct 10-second bar from ticks
                self._construct_10s_bar_from_ticks()
            
        except Exception as e:
            self.logger.error(f"Error processing tick data: {str(e)}")
    
    def _construct_10s_bar_from_ticks(self):
        """Construct 10-second bar from tick data"""
        try:
            # Get recent ticks (last 10 seconds)
            now = datetime.now()
            ten_sec_ago = now - timedelta(seconds=10)
            
            recent_ticks = [tick for tick in self.microstructure_data 
                           if isinstance(tick.get('timestamp'), datetime) and 
                           tick['timestamp'] >= ten_sec_ago]
            
            if not recent_ticks:
                return
            
            # Get prices and volumes
            prices = [tick['price'] for tick in recent_ticks]
            volumes = [tick['volume'] for tick in recent_ticks]
            
            # Construct bar
            bar = {
                'timestamp': ten_sec_ago,
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': sum(volumes)
            }
            
            # Add calculated fields
            bar['price_velocity'] = (bar['close'] - bar['open']) / 10  # 10 seconds
            
            # Add to 10-second buffer
            with self.data_locks['10s']:
                # Convert to DataFrame
                bar_df = pd.DataFrame([bar])
                
                # Append to existing data
                if self.data_buffer['10s'].empty:
                    self.data_buffer['10s'] = bar_df
                else:
                    self.data_buffer['10s'] = pd.concat([self.data_buffer['10s'], bar_df])
                
                # Keep only last N bars
                max_bars = self.config['hf_buffer_size']
                if len(self.data_buffer['10s']) > max_bars:
                    self.data_buffer['10s'] = self.data_buffer['10s'].iloc[-max_bars:]
            
            # Signal new high-frequency data available
            self.hf_data_event.set()
            
        except Exception as e:
            self.logger.error(f"Error constructing 10-second bar: {str(e)}")
    
    def _process_bar_data(self, data):
        """Process bar data for standard timeframes
        
        Args:
            data: Bar data
        """
        try:
            # Extract bar data
            timeframe = data.get('timeframe', '5m')
            
            bar = {
                'timestamp': pd.to_datetime(data['timestamp'], unit='ms'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume']
            }
            
            # Add to buffer
            with self.data_locks[timeframe]:
                # Convert to DataFrame
                bar_df = pd.DataFrame([bar])
                
                # Append to existing data
                if self.data_buffer[timeframe].empty:
                    self.data_buffer[timeframe] = bar_df
                else:
                    self.data_buffer[timeframe] = pd.concat([self.data_buffer[timeframe], bar_df])
                
                # Keep only last N bars
                max_bars = 100  # Lower for standard timeframes
                if len(self.data_buffer[timeframe]) > max_bars:
                    self.data_buffer[timeframe] = self.data_buffer[timeframe].iloc[-max_bars:]
            
            # Signal new data available
            self.new_data_event.set()
            
        except Exception as e:
            self.logger.error(f"Error processing bar data: {str(e)}")
    
    def _is_trading_hours(self):
        """Check if current time is within trading hours
        
        Returns:
            bool: True if within trading hours
        """
        try:
            # Get current time in US Eastern Time (ET)
            now = datetime.now(pytz.timezone('US/Eastern'))
            
            # Check day of week (0 = Monday, 6 = Sunday)
            day_of_week = now.weekday()
            
            # Check if weekend and weekend data is not enabled
            if (day_of_week >= 5) and not self.config['collection_hours']['weekend_data']:
                return False
            
            # Check regular session (9:30 AM - 4:00 PM ET, Monday-Friday)
            is_regular_session = (
                self.config['collection_hours']['regular_session'] and
                day_of_week < 5 and
                now.hour >= 9 and
                (now.hour < 16 or (now.hour == 16 and now.minute == 0))
            )
            
            # Check extended hours (futures can trade nearly 24/5)
            is_extended_hours = (
                self.config['collection_hours']['extended_hours'] and
                day_of_week < 5
            )
            
            return is_regular_session or is_extended_hours
            
        except Exception as e:
            self.logger.error(f"Error checking trading hours: {str(e)}")
            # Default to True to ensure data collection
            return True
    
    def get_historical_data(self, start_date=None, end_date=None, data_type='price', timeframe='5m'):
        """Get historical data from file or API
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            data_type: Data type ('price', 'news', 'economic_calendar')
            timeframe: Data timeframe for price data
            
        Returns:
            pandas.DataFrame: Historical data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Convert to datetime if string
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Check if timeframe is supported
            if data_type == 'price' and timeframe not in self.config['timeframes']:
                self.logger.warning(f"Unsupported timeframe: {timeframe}, using 5m instead")
                timeframe = '5m'
            
            # Check cache first if enabled
            if self.config['use_cache']:
                cached_data = self._get_cached_data(data_type, timeframe, start_date, end_date)
                if cached_data is not None:
                    return cached_data
            
            # Try to get data from file
            file_data = self._get_file_data(data_type, timeframe, start_date, end_date)
            
            if file_data is not None and not file_data.empty:
                # Cache data if enabled
                if self.config['use_cache']:
                    self._cache_data(file_data, data_type, timeframe, start_date, end_date)
                
                return file_data
            
            # Fallback to API if file data not available
            api_data = self._get_api_data(data_type, timeframe, start_date, end_date)
            
            if api_data is not None and not api_data.empty:
                # Cache data if enabled
                if self.config['use_cache']:
                    self._cache_data(api_data, data_type, timeframe, start_date, end_date)
                
                return api_data
            
            self.logger.warning(f"No data available for {data_type} {timeframe} from {start_date} to {end_date}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def _get_cached_data(self, data_type, timeframe, start_date, end_date):
        """Get data from cache
        
        Args:
            data_type: Data type
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas.DataFrame: Cached data or None
        """
        try:
            # Create cache key
            if data_type == 'price':
                cache_key = f"{data_type}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            else:
                cache_key = f"{data_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # Check if cache file exists
            cache_file = os.path.join(self.config['cache_dir'], f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                # Load from cache
                cache_data = pd.read_pickle(cache_file)
                
                self.logger.debug(f"Loaded from cache: {cache_key}")
                
                return cache_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached data: {str(e)}")
            return None
    
    def _cache_data(self, data, data_type, timeframe, start_date, end_date):
        """Cache data to file
        
        Args:
            data: Data to cache
            data_type: Data type
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
        """
        try:
            # Create cache key
            if data_type == 'price':
                cache_key = f"{data_type}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            else:
                cache_key = f"{data_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # Create cache file
            cache_file = os.path.join(self.config['cache_dir'], f"{cache_key}.pkl")
            
            # Save to cache
            data.to_pickle(cache_file)
            
            self.logger.debug(f"Cached data: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")
    
    def _get_file_data(self, data_type, timeframe, start_date, end_date):
        """Get data from file
        
        Args:
            data_type: Data type
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas.DataFrame: File data or None
        """
        try:
            # Get list of data files
            if data_type == 'price':
                file_pattern = f"nq_futures_{timeframe}_*.csv"
            elif data_type == 'news':
                file_pattern = "nq_news_*_*.csv"
            elif data_type == 'economic_calendar':
                file_pattern = "nq_economic_*_*.csv"
            else:
                self.logger.warning(f"Unsupported data type: {data_type}")
                return None
            
            # Get list of files
            data_files = glob.glob(os.path.join(self.config['data_dir'], file_pattern))
            
            if not data_files:
                self.logger.warning(f"No data files found for {data_type} {timeframe}")
                return None
            
            # Read and combine files
            dfs = []
            
            for file in data_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filter by date range
                    if 'timestamp' in df.columns:
                        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    
                    if not df.empty:
                        dfs.append(df)
                
                except Exception as e:
                    self.logger.error(f"Error reading file {file}: {str(e)}")
            
            if not dfs:
                self.logger.warning(f"No data found for {data_type} {timeframe} from {start_date} to {end_date}")
                return None
            
            # Combine dataframes
            combined_df = pd.concat(dfs)
            
            # Remove duplicates
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error getting file data: {str(e)}")
            return None
    
    def _get_api_data(self, data_type, timeframe, start_date, end_date):
        """Get data from API
        
        Args:
            data_type: Data type
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            pandas.DataFrame: API data or None
        """
        try:
            # Only implemented for price data for now
            if data_type != 'price':
                return None
            
            # Check API keys
            if not self.config['api_keys']['primary']:
                self.logger.warning("No API key provided for API data")
                return None
            
            # Construct API URL
            url = f"{self.config['api_endpoints']['primary']}history/NQ/{timeframe}"
            
            # Add API key
            headers = {'Authorization': f"Bearer {self.config['api_keys']['primary']}"}
            
            # Add date range
            params = {
                'start': int(start_date.timestamp() * 1000),
                'end': int(end_date.timestamp() * 1000)
            }
            
            # Make API request
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'bars' not in data or not data['bars']:
                    self.logger.warning(f"No data returned from API for {timeframe} from {start_date} to {end_date}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data['bars'])
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add date column
                if 'timestamp' in df.columns:
                    df['date'] = df['timestamp'].dt.date
                
                return df
            else:
                self.logger.warning(f"Failed to get API data: {response.status_code}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting API data: {str(e)}")
            return None
    
    def get_latest_data(self, data_type='price', timeframe='5m'):
        """Get latest data from memory buffer
        
        Args:
            data_type: Data type ('price', 'news', 'economic_calendar')
            timeframe: Data timeframe for price data
            
        Returns:
            pandas.DataFrame: Latest data
        """
        try:
            # Check if timeframe is supported
            if data_type == 'price' and timeframe not in self.config['timeframes']:
                self.logger.warning(f"Unsupported timeframe: {timeframe}, using 5m instead")
                timeframe = '5m'
            
            # Get data from buffer
            if data_type == 'price':
                with self.data_locks[timeframe]:
                    df = self.data_buffer[timeframe].copy()
                
                return df
            
            # For other data types, read from file
            if data_type == 'news':
                # Combine news from all sources
                dfs = []
                
                for source in self.config['news_sources']:
                    file_path = os.path.join(self.config['data_dir'], f"nq_news_{source}_latest.csv")
                    
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        dfs.append(df)
                
                if not dfs:
                    return None
                
                # Combine and sort by timestamp
                combined_df = pd.concat(dfs)
                
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.sort_values('timestamp', ascending=False)
                
                return combined_df
            
            elif data_type == 'economic_calendar':
                # Combine economic data from all sources
                dfs = []
                
                for source in self.config['economic_calendar_sources']:
                    file_path = os.path.join(self.config['data_dir'], f"nq_economic_{source}_latest.csv")
                    
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        dfs.append(df)
                
                if not dfs:
                    return None
                
                # Combine and sort by timestamp
                combined_df = pd.concat(dfs)
                
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.sort_values('timestamp', ascending=False)
                
                return combined_df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            return None
    
    def wait_for_update(self, timeout=None, high_frequency=False):
        """Wait for new data update
        
        Args:
            timeout: Timeout in seconds (default: None)
            high_frequency: Whether to wait for high-frequency data (default: False)
            
        Returns:
            bool: True if new data available, False if timeout
        """
        try:
            # Select event to wait for
            event = self.hf_data_event if high_frequency else self.new_data_event
            
            # Wait for event
            result = event.wait(timeout)
            
            # Clear event
            event.clear()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error waiting for update: {str(e)}")
            return False