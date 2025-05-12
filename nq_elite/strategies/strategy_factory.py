#!/usr/bin/env python3
"""
Strategy Factory for NQ Alpha Elite

This module provides a factory for creating and managing trading strategies.
"""
import os
import sys
import logging
import importlib
import inspect
from typing import Dict, List, Type, Optional

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nq_alpha_elite import config
from nq_alpha_elite.strategies.base_strategy import BaseStrategy

class StrategyFactory:
    """
    Factory for creating and managing trading strategies
    
    This class provides methods for registering, creating, and managing trading strategies.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the strategy factory
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger("NQAlpha.StrategyFactory")
        
        # Initialize strategy registry
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        
        # Initialize strategy categories
        self.categories = {
            'trend_following': [],
            'mean_reversion': [],
            'breakout': [],
            'volatility': [],
            'pattern': [],
            'multi_timeframe': [],
            'machine_learning': [],
            'hybrid': [],
            'other': []
        }
        
        self.logger.info("Strategy Factory initialized")
    
    def register_strategy(self, strategy_class: Type[BaseStrategy], category: str = 'other') -> None:
        """
        Register a strategy
        
        Args:
            strategy_class (Type[BaseStrategy]): Strategy class
            category (str, optional): Strategy category
        """
        try:
            # Check if strategy inherits from BaseStrategy
            if not issubclass(strategy_class, BaseStrategy):
                self.logger.warning(f"{strategy_class.__name__} does not inherit from BaseStrategy")
                return
            
            # Register strategy
            strategy_name = strategy_class.__name__
            self.strategies[strategy_name] = strategy_class
            
            # Add to category
            if category in self.categories:
                self.categories[category].append(strategy_name)
            else:
                self.categories['other'].append(strategy_name)
            
            self.logger.info(f"Registered strategy {strategy_name} in category {category}")
            
        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy_class.__name__}: {e}")
    
    def create_strategy(self, strategy_name: str, *args, **kwargs) -> Optional[BaseStrategy]:
        """
        Create a strategy instance
        
        Args:
            strategy_name (str): Strategy name
            *args: Positional arguments for strategy constructor
            **kwargs: Keyword arguments for strategy constructor
            
        Returns:
            BaseStrategy: Strategy instance
        """
        try:
            # Check if strategy exists
            if strategy_name not in self.strategies:
                self.logger.warning(f"Strategy {strategy_name} not found")
                return None
            
            # Create strategy instance
            strategy_class = self.strategies[strategy_name]
            strategy = strategy_class(*args, **kwargs)
            
            self.logger.info(f"Created strategy {strategy_name}")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error creating strategy {strategy_name}: {e}")
            return None
    
    def get_strategy_names(self, category: Optional[str] = None) -> List[str]:
        """
        Get strategy names
        
        Args:
            category (str, optional): Strategy category
            
        Returns:
            List[str]: Strategy names
        """
        if category and category in self.categories:
            return self.categories[category]
        elif category:
            self.logger.warning(f"Category {category} not found")
            return []
        else:
            return list(self.strategies.keys())
    
    def get_categories(self) -> List[str]:
        """
        Get strategy categories
        
        Returns:
            List[str]: Strategy categories
        """
        return list(self.categories.keys())
    
    def auto_discover_strategies(self, package_name: str = 'nq_alpha_elite.strategies') -> None:
        """
        Auto-discover strategies in a package
        
        Args:
            package_name (str): Package name
        """
        try:
            # Import package
            package = importlib.import_module(package_name)
            
            # Get package path
            package_path = os.path.dirname(package.__file__)
            
            # Get Python files in package
            python_files = [f[:-3] for f in os.listdir(package_path) 
                           if f.endswith('.py') and f != '__init__.py']
            
            # Import modules
            for module_name in python_files:
                try:
                    # Import module
                    module = importlib.import_module(f"{package_name}.{module_name}")
                    
                    # Get classes in module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if class is a strategy
                        if (issubclass(obj, BaseStrategy) and 
                            obj.__module__ == module.__name__ and 
                            obj != BaseStrategy):
                            
                            # Get category from module name or class attribute
                            category = getattr(obj, 'category', 'other')
                            
                            # Register strategy
                            self.register_strategy(obj, category)
                            
                except (ImportError, AttributeError) as e:
                    self.logger.warning(f"Error importing module {module_name}: {e}")
            
            self.logger.info(f"Auto-discovered strategies in {package_name}")
            
        except Exception as e:
            self.logger.error(f"Error auto-discovering strategies: {e}")

# Create singleton instance
strategy_factory = StrategyFactory()
