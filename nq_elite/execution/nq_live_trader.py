# nq_elite/execution/nq_live_trader.py
import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import queue
import traceback

from nq_elite.data.nq_futures_scraper import NQFuturesScraper
from nq_elite.data.nq_feature_engineering import NQFeaturesProcessor
from nq_elite.models.nq_transformer import NQTransformerModel
from nq_elite.risk.adaptive_risk_manager import AdaptiveRiskManager

class NQLiveTrader:
    """Live trading system for Nasdaq 100 E-mini futures"""
    
    def __init__(self, config=None):
        """Initialize the live trader
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("NQAlpha.LiveTrader")
        
        # Default configuration
        self.config = {
            'execution': {
                'paper_trading': True,
                'live_trading': False,  # Set to True for real trading
                'simulation_mode': False,
                'broker_config': {
                    'api_key': '',
                    'api_secret': '',
                    'account_id': '',
                    'base_url': '',
                    'paper_base_url': ''
                },
                'order_types': {
                    'entry': 'market',  # market, limit, stop
                    'exit': 'market',
                    'stop_loss': 'stop',
                    'take_profit': 'limit',
                    'trailing_stop': 'trailing_stop'
                },
                'execution_algo': 'twap',  # vwap, twap, or direct
                'max_slippage_bps': 5,  # 5 basis points max slippage
                'retry_failed_orders': True,
                'max_retries': 3,
                'retry_delay_seconds': 5,
                'cancel_after_seconds': 60,
                'trading_hours': {
                    'enabled': True,
                    'timezone': 'America/Chicago',
                    'regular_session_open': '08:30',
                    'regular_session_close': '15:15',
                    'extended_hours': True,
                    'extended_open': '17:00',  # Previous day
                    'extended_close': '16:00',
                    'trade_settlements': True
                },
                'market_data_feed': 'broker_api'  # broker_api, websocket, or scraper
            },
            'strategy': {
                'name': 'NQAlphaElite',
                'timeframe': '5m',
                'data_lookback_periods': 100,
                'model_path': 'models/nq_5m_model',
                'processor_path': 'models/output/nq_5m_feature_processor.pkl',
                'signal_threshold': 0.6,  # Minimum probability to generate signal
                'confidence_scaling': True,
                'trade_frequency': 'every_bar',  # tick, every_bar, or fixed_interval
                'interval_minutes': 5,  # For fixed_interval
                'entry_logic': 'model_only',  # model_only, model_with_filters, ensemble
                'exit_logic': {
                    'use_model': True,  # Use model signals for exits
                    'use_stops': True,  # Use stop-loss and take-profit
                    'use_time_stops': True,  # Use time-based stops
                    'max_hold_periods': 24  # In bars (e.g., 24 5-minute bars = 2 hours)
                },
                'filters': {
                    'volatility_filter': True,
                    'trend_filter': True,
                    'time_filter': True,
                    'volume_filter': True,
                    'news_filter': True
                },
                'multi_timeframe': {
                    'enabled': True,
                    'timeframes': ['5m', '1h', 'daily'],
                    'confirmation_method': 'majority',  # majority, weighted, or strict
                    'weights': {'5m': 0.5, '1h': 0.3, 'daily': 0.2}
                }
            },
            'risk': {
                'account_config': {
                    'initial_capital': 100000,
                    'max_leverage': 1.5,
                    'max_drawdown_pct': 0.20,
                    'max_daily_drawdown_pct': 0.05
                },
                'position_sizing': {
                    'method': 'risk_parity',
                    'max_position_pct': 0.20,
                    'min_position_pct': 0.01,
                    'target_risk_pct': 0.01
                },
                'stop_loss': {
                    'enabled': True,
                    'fixed_stop_loss_pct': 0.02,
                    'trailing_stop_loss_pct': 0.03,
                    'volatility_based_stops': True,
                    'atr_multiplier': 3.0
                },
                'take_profit': {
                    'enabled': True,
                    'fixed_take_profit_pct': 0.05,
                    'volatility_based_tp': True,
                    'atr_multiplier': 5.0
                },
                'circuit_breakers': {
                    'enabled': True,
                    'loss_per_trade_pct': 0.05,
                    'loss_per_day_pct': 0.07,
                    'consecutive_losses': 5
                }
            },
            'monitoring': {
                'log_level': 'INFO',
                'log_trades': True,
                'log_signals': True,
                'save_state': True,
                'state_update_interval': 60,  # seconds
                'state_file': 'trader_state.json',
                'health_checks': {
                    'data_feed': True,
                    'model_latency': True,
                    'broker_connection': True,
                    'performance_metrics': True
                },
                'alerts': {
                    'enabled': True,
                    'email': '',
                    'sms': '',
                    'webhook': '',
                    'levels': ['error', 'warning', 'info']
                },
                'metrics_tracking': {
                    'enabled': True,
                    'track_pnl': True,
                    'track_drawdown': True,
                    'track_win_rate': True,
                    'track_model_accuracy': True,
                    'track_prediction_quality': True
                }
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(self.config, config)
        
        # Initialize components
        self.scraper = None
        self.feature_processor = None
        self.model = None
        self.risk_manager = None
        
        # Initialize state
        self.running = False
        self.state = {
            'status': 'initialized',
            'last_update': datetime.now().isoformat(),
            'last_signal_time': None,
            'last_trade_time': None,
            'signals': {},
            'last_prices': {},
            'daily_stats': {
                'signals': 0,
                'trades': 0,
                'pnl': 0.0
            },
            'cumulative_stats': {
                'signals': 0,
                'trades': 0,
                'pnl': 0.0,
                'win_count': 0,
                'loss_count': 0
            },
            'errors': []
        }
        
        # Initialize thread and queues
        self.trader_thread = None
        self.stop_event = threading.Event()
        self.signal_queue = queue.Queue()
        self.order_queue = queue.Queue()
        self.data_buffer = {
            '5m': [],
            '15m': [],
            '1h': [],
            '4h': [],
            'daily': []
        }
        
        # Load components
        self._load_components()
        
        self.logger.info("NQ Live Trader initialized")
    
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
    
    def _load_components(self):
        """Load scraper, feature processor, model, and risk manager"""
        try:
            # Initialize data scraper
            self.scraper = NQFuturesScraper()
            
            # Load feature processor
            processor_path = self.config['strategy']['processor_path']
            if os.path.exists(processor_path):
                import joblib
                self.feature_processor = joblib.load(processor_path)
                self.logger.info(f"Feature processor loaded from {processor_path}")
            else:
                self.feature_processor = NQFeaturesProcessor()
                self.logger.warning(f"Feature processor not found at {processor_path}, using default")
            
            # Load model
            model_path = self.config['strategy']['model_path']
            self.model = NQTransformerModel()
            
            if os.path.exists(model_path):
                if self.model.load(model_path):
                    self.logger.info(f"Model loaded from {model_path}")
                else:
                    self.logger.error(f"Failed to load model from {model_path}")
            else:
                self.logger.error(f"Model not found at {model_path}")
            
            # Initialize risk manager
            risk_config = self.config['risk']
            self.risk_manager = AdaptiveRiskManager(risk_config)
            
        except Exception as e:
            self.logger.error(f"Error loading components: {str(e)}")
            traceback.print_exc()
    
    def start(self):
        """Start the trader
        
        Returns:
            bool: Success flag
        """
        try:
            if self.running:
                self.logger.warning("Trader already running")
                return False
            
            self.logger.info("Starting NQ Live Trader")
            
            # Start scraper
            if not self.scraper.start():
                self.logger.error("Failed to start data scraper")
                return False
            
            # Set running flag
            self.running = True
            self.stop_event.clear()
            
            # Start trader thread
            self.trader_thread = threading.Thread(
                target=self._trader_loop,
                name="NQTraderThread",
                daemon=True
            )
            self.trader_thread.start()
            
            # Update state
            self.state['status'] = 'running'
            self.state['last_update'] = datetime.now().isoformat()
            
            # Save initial state
            if self.config['monitoring']['save_state']:
                self._save_state()
            
            self.logger.info("NQ Live Trader started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting trader: {str(e)}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the trader
        
        Returns:
            bool: Success flag
        """
        try:
            if not self.running:
                self.logger.warning("Trader not running")
                return False
            
            self.logger.info("Stopping NQ Live Trader")
            
            # Set stop event and clear running flag
            self.stop_event.set()
            self.running = False
            
            # Wait for thread to exit
            if self.trader_thread and self.trader_thread.is_alive():
                self.trader_thread.join(timeout=10)
            
            # Stop scraper
            if self.scraper and self.scraper.running:
                self.scraper.stop()
            
            # Update state
            self.state['status'] = 'stopped'
            self.state['last_update'] = datetime.now().isoformat()
            
            # Save final state
            if self.config['monitoring']['save_state']:
                self._save_state()
            
            self.logger.info("NQ Live Trader stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trader: {str(e)}")
            self.running = False
            return False
    
    def _trader_loop(self):
        """Main trader loop"""
        try:
            last_data_check = 0
            last_state_save = 0
            
            while self.running and not self.stop_event.is_set():
                current_time = time.time()
                
                # Check if trading hours
                if self.config['execution']['trading_hours']['enabled'] and not self._is_trading_hours():
                    # Outside trading hours, sleep and continue
                    time.sleep(5)
                    continue
                
                # Check for new data (every 10 seconds)
                if current_time - last_data_check >= 10:
                    self._check_new_data()
                    last_data_check = current_time
                
                # Process signals in queue
                while not self.signal_queue.empty():
                    signal = self.signal_queue.get()
                    self._process_signal(signal)
                
                # Process orders in queue
                while not self.order_queue.empty():
                    order = self.order_queue.get()
                    self._execute_order(order)
                
                # Save state if configured
                if self.config['monitoring']['save_state'] and current_time - last_state_save >= self.config['monitoring']['state_update_interval']:
                    self._save_state()
                    last_state_save = current_time
                
                # Sleep briefly
                time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Error in trader loop: {str(e)}")
            traceback.print_exc()
            self.running = False
    
    def _check_new_data(self):
        """Check for new market data and update trading signals"""
        try:
            # Get timeframe from strategy config
            timeframe = self.config['strategy']['timeframe']
            
            # Get latest data from scraper
            latest_data = self.scraper.get_latest_data(data_type='price', timeframe=timeframe)
            
            if latest_data is None or len(latest_data) == 0:
                self.logger.warning(f"No data available for {timeframe}")
                return
            
            # Get last price
            symbol = 'NQ'  # Nasdaq 100 E-mini
            
            if 'close' in latest_data.columns:
                last_price = latest_data['close'].iloc[-1]
            elif 'Close' in latest_data.columns:
                last_price = latest_data['Close'].iloc[-1]
            else:
                self.logger.warning(f"Close price not found in data")
                return
            
            # Store last price
            self.state['last_prices'][symbol] = last_price
            
            # Check if we need to update signals based on trade frequency
            trade_frequency = self.config['strategy']['trade_frequency']
            
            if trade_frequency == 'every_bar':
                # Generate signals on every new bar
                self._generate_trading_signals(latest_data)
            elif trade_frequency == 'fixed_interval':
                # Generate signals on fixed time interval
                interval_minutes = self.config['strategy']['interval_minutes']
                last_signal_time = self.state.get('last_signal_time')
                
                if last_signal_time is None:
                    # No previous signal, generate one
                    self._generate_trading_signals(latest_data)
                else:
                    # Check if interval has passed
                    last_signal_datetime = datetime.fromisoformat(last_signal_time)
                    if (datetime.now() - last_signal_datetime) >= timedelta(minutes=interval_minutes):
                        self._generate_trading_signals(latest_data)
            
            # Update risk manager with new price data
            if self.risk_manager is not None:
                self.risk_manager.update_prices(symbol, latest_data)
            
        except Exception as e:
            self.logger.error(f"Error checking new data: {str(e)}")
            self.state['errors'].append({
                'time': datetime.now().isoformat(),
                'type': 'data_error',
                'message': str(e)
            })
    
    def _generate_trading_signals(self, price_data):
        """Generate trading signals from price data
        
        Args:
            price_data: Latest price data
        """
        try:
            symbol = 'NQ'  # Nasdaq 100 E-mini
            
            # Get news and economic data if available
            news_data = self.scraper.get_latest_data(data_type='news')
            economic_data = self.scraper.get_latest_data(data_type='economic_calendar')
            
            # Process data to create features
            features_df, targets_df = self.feature_processor.process_data(
                price_data=price_data,
                news_data=news_data,
                economic_data=economic_data,
                timeframe=self.config['strategy']['timeframe']
            )
            
            if features_df is None:
                self.logger.warning("Failed to generate features")
                return
            
            # Make predictions
            predictions = self.model.predict(features_df)
            
            if predictions is None:
                self.logger.warning("Model failed to generate predictions")
                return
            
            # Process predictions
            signals = self._process_predictions(symbol, predictions, features_df)
            
            # Apply filters if configured
            if self.config['strategy']['entry_logic'] == 'model_with_filters':
                signals = self._apply_filters(signals, features_df)
            
            # Multi-timeframe confirmation if enabled
            if self.config['strategy']['multi_timeframe']['enabled']:
                signals = self._apply_multi_timeframe_confirmation(signals, features_df)
            
            # Enqueue valid signals
            if signals and signals.get('valid', False):
                self.signal_queue.put(signals)
                
                # Update state
                self.state['last_signal_time'] = datetime.now().isoformat()
                self.state['signals'][symbol] = signals
                self.state['daily_stats']['signals'] += 1
                self.state['cumulative_stats']['signals'] += 1
                
                # Log signal
                if self.config['monitoring']['log_signals']:
                    self.logger.info(f"Signal generated: {signals}")
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            traceback.print_exc()
            self.state['errors'].append({
                'time': datetime.now().isoformat(),
                'type': 'signal_error',
                'message': str(e)
            })
    
    def _process_predictions(self, symbol, predictions, features_df):
        """Process model predictions into trading signals
        
        Args:
            symbol: Trading symbol
            predictions: Model predictions
            features_df: Features DataFrame
            
        Returns:
            dict: Trading signals
        """
        try:
            # Initialize signals
            signals = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'valid': False,
                'direction': None,
                'confidence': 0.0,
                'signal_strength': 0.0,
                'entry_price': self.state['last_prices'].get(symbol, 0),
                'raw_predictions': {},
                'metadata': {}
            }
            
            # Get primary target (e.g., 'direction_5m')
            primary_targets = [t for t in self.config['strategy']['model_config']['prediction_targets'] 
                              if t.startswith('direction_')]
            
            if not primary_targets:
                self.logger.warning("No direction prediction targets found")
                return signals
            
            primary_target = primary_targets[0]
            
            # Get prediction for primary target
            if primary_target in predictions:
                if 'probabilities' in predictions[primary_target]:
                    probs = predictions[primary_target]['probabilities']
                    
                    # For binary classification, probability is for class 1 (up)
                    if len(probs.shape) == 1 or probs.shape[1] == 1:
                        up_prob = probs[0] if len(probs.shape) == 1 else probs[0, 0]
                        down_prob = 1.0 - up_prob
                        
                        # Determine signal direction
                        threshold = self.config['strategy']['signal_threshold']
                        
                        if up_prob >= threshold:
                            signals['direction'] = 'long'
                            signals['confidence'] = up_prob
                            signals['valid'] = True
                        elif down_prob >= threshold:
                            signals['direction'] = 'short'
                            signals['confidence'] = down_prob
                            signals['valid'] = True
                        
                        # Store raw predictions
                        signals['raw_predictions'][primary_target] = {
                            'up_probability': float(up_prob),
                            'down_probability': float(down_prob)
                        }
                    else:
                        # Multi-class, not currently handled
                        pass
            
            # Get additional return predictions if available
            return_targets = [t for t in self.config['strategy']['model_config']['prediction_targets'] 
                             if t.startswith('return_')]
            
            for target in return_targets:
                if target in predictions:
                    # Store return prediction
                    signals['raw_predictions'][target] = float(predictions[target][0])
            
            # If we have a valid signal, calculate signal strength
            if signals['valid']:
                # Signal strength is a function of confidence and possibly other factors
                signals['signal_strength'] = signals['confidence']
                
                # Adjust based on return prediction if available
                main_return_target = f"return_{primary_target.split('_')[1]}"
                if main_return_target in signals['raw_predictions']:
                    predicted_return = signals['raw_predictions'][main_return_target]
                    
                    # Scale signal strength by normalized predicted return
                    if signals['direction'] == 'long' and predicted_return > 0:
                        # Scale up for positive predicted return on long
                        signals['signal_strength'] *= (1 + min(predicted_return * 20, 0.5))
                    elif signals['direction'] == 'short' and predicted_return < 0:
                        # Scale up for negative predicted return on short
                        signals['signal_strength'] *= (1 + min(abs(predicted_return) * 20, 0.5))
                    elif (signals['direction'] == 'long' and predicted_return < 0) or \
                         (signals['direction'] == 'short' and predicted_return > 0):
                        # Contradictory signals, reduce strength
                        signals['signal_strength'] *= (1 - min(abs(predicted_return) * 20, 0.5))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing predictions: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def _apply_filters(self, signals, features_df):
        """Apply filters to trading signals
        
        Args:
            signals: Trading signals
            features_df: Features DataFrame
            
        Returns:
            dict: Filtered signals
        """
        try:
            if not signals['valid']:
                return signals
            
            filters = self.config['strategy']['filters']
            filtered_signals = signals.copy()
            
            # Get latest bar
            latest_bar = features_df.iloc[-1]
            
            # Initialize filter status
            filtered_signals['metadata']['filters'] = {}
            
            # 1. Volatility filter
            if filters['volatility_filter']:
                volatility_ok = True
                
                # Check if we're in extreme volatility regimes
                if 'volatility_regime' in latest_bar:
                    regime = latest_bar['volatility_regime']
                    
                    # Skip high volatility for safety
                    if regime == 2:  # High volatility
                        volatility_ok = False
                
                # Check volatility trend
                if 'volatility_trend' in latest_bar:
                    vol_trend = latest_bar['volatility_trend']
                    
                    # Be cautious in rising volatility
                    if vol_trend == 1 and signals['direction'] == 'long':
                        volatility_ok = False
                
                filtered_signals['metadata']['filters']['volatility'] = volatility_ok
                
                if not volatility_ok:
                    filtered_signals['valid'] = False
                    return filtered_signals
            
            # 2. Trend filter
            if filters['trend_filter']:
                trend_ok = True
                
                # Check market trend
                if 'trend_regime' in latest_bar:
                    trend = latest_bar['trend_regime']
                    
                    # Align with trend
                    if (signals['direction'] == 'long' and trend < 0) or \
                       (signals['direction'] == 'short' and trend > 0):
                        trend_ok = False
                
                filtered_signals['metadata']['filters']['trend'] = trend_ok
                
                if not trend_ok:
                    filtered_signals['valid'] = False
                    return filtered_signals
            
            # 3. Time filter
            if filters['time_filter']:
                time_ok = True
                
                # Avoid trading near settlement
                if 'date' in features_df.columns:
                    current_time = pd.to_datetime(features_df['date'].iloc[-1])
                elif 'datetime' in features_df.columns:
                    current_time = pd.to_datetime(features_df['datetime'].iloc[-1])
                else:
                    current_time = datetime.now()
                
                # Check if near market close
                market_close = pd.Timestamp(
                    year=current_time.year,
                    month=current_time.month,
                    day=current_time.day,
                    hour=15,  # 3 PM in exchange timezone
                    minute=10  # 3:10 PM
                )
                
                if current_time.hour == market_close.hour and abs(current_time.minute - market_close.minute) < 15:
                    # Within 15 minutes of settlement
                    time_ok = False
                
                filtered_signals['metadata']['filters']['time'] = time_ok
                
                if not time_ok:
                    filtered_signals['valid'] = False
                    return filtered_signals
            
            # 4. Volume filter
            if filters['volume_filter']:
                volume_ok = True
                
                # Check for adequate volume
                if 'volume' in features_df.columns:
                    current_volume = features_df['volume'].iloc[-1]
                    avg_volume = features_df['volume'].rolling(20).mean().iloc[-1]
                    
                    if current_volume < 0.5 * avg_volume:
                        # Volume is too low
                        volume_ok = False
                
                filtered_signals['metadata']['filters']['volume'] = volume_ok
                
                if not volume_ok:
                    filtered_signals['valid'] = False
                    return filtered_signals
            
            # 5. News filter
            if filters['news_filter']:
                news_ok = True
                
                # Check for extreme news sentiment
                if 'news_sentiment' in latest_bar:
                    sentiment = latest_bar['news_sentiment']
                    
                    # Be cautious with extreme sentiment
                    if abs(sentiment) > 0.7:
                        # Strong sentiment, check alignment with signal
                        if (signals['direction'] == 'long' and sentiment < -0.7) or \
                           (signals['direction'] == 'short' and sentiment > 0.7):
                            news_ok = False
                
                filtered_signals['metadata']['filters']['news'] = news_ok
                
                if not news_ok:
                    filtered_signals['valid'] = False
                    return filtered_signals
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
            # Return original signals on error
            return signals
    
    def _apply_multi_timeframe_confirmation(self, signals, features_df):
        """Apply multi-timeframe confirmation to signals
        
        Args:
            signals: Trading signals
            features_df: Features DataFrame
            
        Returns:
            dict: Confirmed signals
        """
        try:
            if not signals['valid'] or not self.config['strategy']['multi_timeframe']['enabled']:
                return signals
            
            # Get multi-timeframe configuration
            mt_config = self.config['strategy']['multi_timeframe']
            timeframes = mt_config['timeframes']
            confirmation_method = mt_config['confirmation_method']
            
            # Skip if we're only using one timeframe
            primary_tf = self.config['strategy']['timeframe']
            if len(timeframes) <= 1 or primary_tf not in timeframes:
                return signals
            
            # Get signals for each timeframe
            tf_signals = {primary_tf: signals['direction']}
            
            # Check other timeframes
            for tf in timeframes:
                if tf == primary_tf:
                    continue
                
                # Get data for this timeframe
                tf_data = self.scraper.get_latest_data(data_type='price', timeframe=tf)
                
                if tf_data is None or len(tf_data) == 0:
                    # No data, skip this timeframe
                    continue
                
                # Process data to create features
                tf_features, _ = self.feature_processor.process_data(
                    price_data=tf_data,
                    timeframe=tf
                )
                
                if tf_features is None:
                    # Failed to process, skip this timeframe
                    continue
                
                # Make predictions
                tf_predictions = self.model.predict(tf_features)
                
                if tf_predictions is None:
                    # Failed to predict, skip this timeframe
                    continue
                
                # Get prediction direction
                primary_targets = [t for t in self.config['strategy']['model_config']['prediction_targets'] 
                                  if t.startswith('direction_')]
                
                if not primary_targets:
                    continue
                
                primary_target = primary_targets[0]
                
                if primary_target in tf_predictions and 'probabilities' in tf_predictions[primary_target]:
                    probs = tf_predictions[primary_target]['probabilities']
                    
                    # For binary classification, probability is for class 1 (up)
                    if len(probs.shape) == 1 or probs.shape[1] == 1:
                        up_prob = probs[0] if len(probs.shape) == 1 else probs[0, 0]
                        down_prob = 1.0 - up_prob
                        
                        # Get direction
                        threshold = self.config['strategy']['signal_threshold']
                        
                        if up_prob >= threshold:
                            tf_signals[tf] = 'long'
                        elif down_prob >= threshold:
                            tf_signals[tf] = 'short'
                        else:
                            tf_signals[tf] = 'neutral'
            
            # Apply confirmation method
            confirmed = False
            
            if confirmation_method == 'majority':
                # Count signals by direction
                long_count = sum(1 for d in tf_signals.values() if d == 'long')
                short_count = sum(1 for d in tf_signals.values() if d == 'short')
                
                if signals['direction'] == 'long' and long_count > short_count:
                    confirmed = True
                elif signals['direction'] == 'short' and short_count > long_count:
                    confirmed = True
            
            elif confirmation_method == 'weighted':
                # Weighted average by timeframe
                weights = mt_config['weights']
                
                # Calculate weighted score (-1 for short, 0 for neutral, 1 for long)
                score = 0.0
                for tf, direction in tf_signals.items():
                    if tf in weights:
                        if direction == 'long':
                            score += weights[tf]
                        elif direction == 'short':
                            score -= weights[tf]
                
                # Confirm if score matches direction
                if signals['direction'] == 'long' and score > 0:
                    confirmed = True
                elif signals['direction'] == 'short' and score < 0:
                    confirmed = True
            
            elif confirmation_method == 'strict':
                # All timeframes must agree
                if signals['direction'] == 'long':
                    confirmed = all(d in ('long', 'neutral') for d in tf_signals.values())
                elif signals['direction'] == 'short':
                    confirmed = all(d in ('short', 'neutral') for d in tf_signals.values())
            
            # Update signals
            signals['valid'] = confirmed
            signals['metadata']['multi_timeframe'] = {
                'confirmed': confirmed,
                'timeframe_signals': tf_signals
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error applying multi-timeframe confirmation: {str(e)}")
            # Return original signals on error
            return signals
    
    def _process_signal(self, signal):
        """Process trading signal and generate orders
        
        Args:
            signal: Trading signal
        """
        try:
            if not signal['valid']:
                return
            
            symbol = signal['symbol']
            direction = signal['direction']
            confidence = signal['confidence']
            
            # Get current position
            current_position = self.risk_manager.get_position_info(symbol)
            
            # Get current price
            current_price = self.state['last_prices'].get(symbol, signal['entry_price'])
            
            # Determine if we should enter a new position
            if not current_position:
                # No current position, check if we should enter
                
                # Calculate position size
                price_data = self.scraper.get_latest_data(data_type='price', timeframe=self.config['strategy']['timeframe'])
                
                position_size_info = self.risk_manager.calculate_position_size(
                    symbol, 
                    signal['signal_strength'], 
                    confidence, 
                    price_data
                )
                
                if position_size_info['position_size'] <= 0:
                    self.logger.info(f"Signal rejected due to zero position size: {position_size_info['reason']}")
                    return
                
                # Calculate stops
                stop_loss = None
                take_profit = None
                
                if self.config['risk']['stop_loss']['enabled']:
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        symbol, 
                        direction, 
                        current_price, 
                        price_data
                    )
                
                if self.config['risk']['take_profit']['enabled']:
                    take_profit = self.risk_manager.calculate_take_profit(
                        symbol, 
                        direction, 
                        current_price, 
                        price_data
                    )
                
                # Create entry order
                entry_order = {
                    'symbol': symbol,
                    'order_type': self.config['execution']['order_types']['entry'],
                    'direction': direction,
                    'quantity': position_size_info['position_size'],
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal_time': signal['timestamp'],
                    'order_time': datetime.now().isoformat(),
                    'confidence': confidence
                }
                
                # Enqueue order
                self.order_queue.put(entry_order)
                
                self.logger.info(f"Entry order created: {direction} {position_size_info['position_size']} {symbol} @ {current_price}")
            
            else:
                # Existing position, check if we should exit or reverse
                current_direction = current_position.get('position_type')
                
                if current_direction != direction:
                    # Signal opposite to current position, exit
                    
                    # Create exit order
                    exit_order = {
                        'symbol': symbol,
                        'order_type': self.config['execution']['order_types']['exit'],
                        'direction': 'exit',
                        'quantity': current_position.get('position_size', 0),
                        'price': current_price,
                        'signal_time': signal['timestamp'],
                        'order_time': datetime.now().isoformat(),
                        'confidence': confidence
                    }
                    
                    # Enqueue order
                    self.order_queue.put(exit_order)
                    
                    self.logger.info(f"Exit order created: exit {current_position.get('position_size', 0)} {symbol} @ {current_price}")
                    
                    # If trading logic allows position reversal, create entry order too
                    if self.config['strategy']['entry_logic'] != 'model_only':
                        # Calculate position size for new position
                        price_data = self.scraper.get_latest_data(data_type='price', timeframe=self.config['strategy']['timeframe'])
                        
                        position_size_info = self.risk_manager.calculate_position_size(
                            symbol, 
                            signal['signal_strength'], 
                            confidence, 
                            price_data
                        )
                        
                        if position_size_info['position_size'] <= 0:
                            return
                        
                        # Calculate stops
                        stop_loss = None
                        take_profit = None
                        
                        if self.config['risk']['stop_loss']['enabled']:
                            stop_loss = self.risk_manager.calculate_stop_loss(
                                symbol, 
                                direction, 
                                current_price, 
                                price_data
                            )
                        
                        if self.config['risk']['take_profit']['enabled']:
                            take_profit = self.risk_manager.calculate_take_profit(
                                symbol, 
                                direction, 
                                current_price, 
                                price_data
                            )
                        
                        # Create entry order for new direction
                        entry_order = {
                            'symbol': symbol,
                            'order_type': self.config['execution']['order_types']['entry'],
                            'direction': direction,
                            'quantity': position_size_info['position_size'],
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_time': signal['timestamp'],
                            'order_time': datetime.now().isoformat(),
                            'confidence': confidence
                        }
                        
                        # Enqueue order
                        self.order_queue.put(entry_order)
                        
                        self.logger.info(f"Reversal entry order created: {direction} {position_size_info['position_size']} {symbol} @ {current_price}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            traceback.print_exc()
            self.state['errors'].append({
                'time': datetime.now().isoformat(),
                'type': 'signal_processing_error',
                'message': str(e)
            })
    
    def _execute_order(self, order):
        """Execute trading order
        
        Args:
            order: Order details
        """
        try:
            # Check if paper trading or live trading
            if self.config['execution']['paper_trading']:
                # Paper trading execution
                self._execute_paper_order(order)
            elif self.config['execution']['live_trading']:
                # Live trading execution
                self._execute_live_order(order)
            else:
                # Simulation mode or disabled
                self.logger.info(f"Order simulated (not executed): {order}")
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            traceback.print_exc()
            self.state['errors'].append({
                'time': datetime.now().isoformat(),
                'type': 'order_execution_error',
                'message': str(e)
            })
    
    def _execute_paper_order(self, order):
        """Execute order in paper trading mode
        
        Args:
            order: Order details
        """
        try:
            symbol = order['symbol']
            order_type = order['order_type']
            direction = order['direction']
            quantity = order['quantity']
            price = order['price']
            
            # Apply simulated slippage
            if order_type == 'market':
                # Apply slippage to price
                max_slippage_bps = self.config['execution']['max_slippage_bps']
                slippage_factor = max_slippage_bps / 10000  # Convert bps to decimal
                
                if direction == 'long':
                    # Buy at slightly higher price
                    execution_price = price * (1 + slippage_factor)
                elif direction == 'short':
                    # Sell at slightly lower price
                    execution_price = price * (1 - slippage_factor)
                else:  # exit
                    # Use midpoint
                    execution_price = price
            else:
                # Limit or stop orders execute at specified price
                execution_price = price
            
            # Create execution record
            execution = {
                'symbol': symbol,
                'order_type': order_type,
                'direction': direction,
                'quantity': quantity,
                'requested_price': price,
                'execution_price': execution_price,
                'order_time': order['order_time'],
                'execution_time': datetime.now().isoformat(),
                'status': 'filled',
                'paper_trade': True
            }
            
            # Apply order to risk manager
            if direction == 'exit':
                # Exit position
                self.risk_manager.update_position(symbol, {'position_size': 0}, execution_price)
                
                self.logger.info(f"Paper trade executed: exit {quantity} {symbol} @ {execution_price}")
            else:
                # Enter position
                position_info = {
                    'position_type': direction,
                    'position_size': quantity,
                    'entry_price': execution_price,
                    'stop_loss': order.get('stop_loss'),
                    'take_profit': order.get('take_profit'),
                    'entry_time': datetime.now()
                }
                
                self.risk_manager.update_position(symbol, position_info, execution_price)
                
                self.logger.info(f"Paper trade executed: {direction} {quantity} {symbol} @ {execution_price}")
            
            # Update state
            self.state['last_trade_time'] = datetime.now().isoformat()
            self.state['daily_stats']['trades'] += 1
            self.state['cumulative_stats']['trades'] += 1
            
            # Log trade
            if self.config['monitoring']['log_trades']:
                log_msg = f"Paper trade executed: {direction} {quantity} {symbol} @ {execution_price}"
                if direction == 'long' or direction == 'short':
                    log_msg += f" SL: {order.get('stop_loss')}, TP: {order.get('take_profit')}"
                
                self.logger.info(log_msg)
            
        except Exception as e:
            self.logger.error(f"Error executing paper order: {str(e)}")
    
    def _execute_live_order(self, order):
        """Execute order in live trading mode
        
        Args:
            order: Order details
        """
        try:
            self.logger.warning("Live trading not implemented yet")
            # In a real implementation, this would connect to a broker API
            # and execute the trade, then handle order status updates
            
            # For now, log the order and treat it as paper trade
            self.logger.info(f"Would execute live order: {order}")
            
            # Execute as paper trade instead
            self._execute_paper_order(order)
            
        except Exception as e:
            self.logger.error(f"Error executing live order: {str(e)}")
    
    def _is_trading_hours(self):
        """Check if current time is within trading hours
        
        Returns:
            bool: True if within trading hours
        """
        try:
            # Get trading hours config
            trading_hours = self.config['execution']['trading_hours']
            
            if not trading_hours['enabled']:
                # Trading hours check disabled
                return True
            
            # Get timezone
            import pytz
            import dateutil.parser
            
            timezone_str = trading_hours['timezone']
            timezone = pytz.timezone(timezone_str)
            
            # Get current time in exchange timezone
            current_time = datetime.now(timezone)
            
            # Parse session times
            regular_open = dateutil.parser.parse(trading_hours['regular_session_open']).time()
            regular_close = dateutil.parser.parse(trading_hours['regular_session_close']).time()
            
            # Check regular session
            current_time_only = current_time.time()
            
            if regular_open <= current_time_only <= regular_close:
                return True
            
            # Check extended hours if enabled
            if trading_hours['extended_hours']:
                extended_open = dateutil.parser.parse(trading_hours['extended_open']).time()
                extended_close = dateutil.parser.parse(trading_hours['extended_close']).time()
                
                # Check if within extended hours
                # Note: extended open can be after extended close if spanning midnight
                if extended_open <= extended_close:
                    # Normal case
                    if extended_open <= current_time_only <= extended_close:
                        return True
                else:
                    # Spans midnight
                    if current_time_only >= extended_open or current_time_only <= extended_close:
                        return True
            
            # Not in trading hours
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trading hours: {str(e)}")
            # Default to allowing trading on error
            return True
    
    def _save_state(self):
        """Save trader state to file"""
        try:
            # Update last update time
            self.state['last_update'] = datetime.now().isoformat()
            
            # Save state to file
            state_file = self.config['monitoring']['state_file']
            
            with open(state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def get_state(self):
        """Get current trader state
        
        Returns:
            dict: Trader state
        """
        return self.state
    
    def get_risk_metrics(self):
        """Get risk metrics
        
        Returns:
            dict: Risk metrics
        """
        if self.risk_manager:
            return self.risk_manager.get_risk_metrics()
        return {}
    
    def get_positions(self):
        """Get current positions
        
        Returns:
            dict: Current positions
        """
        if self.risk_manager:
            return self.risk_manager.get_position_info()
        return {}