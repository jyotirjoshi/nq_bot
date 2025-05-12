#!/usr/bin/env python3
# nq_elite/main.py
import os
import sys
import argparse
import logging
import json
import time
import signal
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core components
from nq_alpha_elite.core.master_controller import MasterController

def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Create banner
    logger = logging.getLogger("NQAlpha")
    banner = """
    ███╗   ██╗ ██████╗      █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗     ███████╗██╗     ██╗████████╗███████╗
    ████╗  ██║██╔═══██╗    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗    ██╔════╝██║     ██║╚══██╔══╝██╔════╝
    ██╔██╗ ██║██║   ██║    ███████║██║     ██████╔╝███████║███████║    █████╗  ██║     ██║   ██║   █████╗  
    ██║╚██╗██║██║▄▄ ██║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║    ██╔══╝  ██║     ██║   ██║   ██╔══╝  
    ██║ ╚████║╚██████╔╝    ██║  ██║███████╗██║     ██║  ██║██║  ██║    ███████╗███████╗██║   ██║   ███████╗
    ╚═╝  ╚═══╝ ╚══▀▀═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝
                                                                                                v5.0.0
    """
    
    logger.info(banner)
    logger.info(f"NQ Alpha Elite Trading System - Version 5.0.0")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log level: {log_level}")

def parse_arguments():
    """Parse command line arguments
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='NQ Alpha Elite Trading System')
    
    # System configuration
    parser.add_argument('--config', type=str, default='config/system_config.json',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file')
    
    # Operating mode
    parser.add_argument('--mode', type=str, default='live',
                        choices=['live', 'paper', 'backtest', 'optimize'],
                        help='Operating mode')
    
    # Backtest/optimization options
    parser.add_argument('--start-date', type=str,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str,
                        help='Comma-separated list of symbols to trade')
    
    # Resource limits
    parser.add_argument('--memory-limit', type=float, default=0,
                        help='Memory limit in GB (0 for no limit)')
    parser.add_argument('--cpu-limit', type=int, default=0,
                        help='CPU core limit (0 for no limit)')
    
    # Advanced options
    parser.add_argument('--diagnostics', action='store_true',
                        help='Enable detailed diagnostics')
    parser.add_argument('--safe-mode', action='store_true',
                        help='Run in safe mode with additional safeguards')
    parser.add_argument('--enable-quantum', action='store_true',
                        help='Enable quantum computing features')
    parser.add_argument('--api-only', action='store_true',
                        help='Run only the API server')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable web dashboard')
    
    # Commands
    parser.add_argument('command', nargs='?', default='start',
                        choices=['start', 'status', 'stop', 'restart', 'version', 'shell'],
                        help='Command to execute')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Check if configuration file exists
        if not os.path.isfile(config_path):
            logger = logging.getLogger("NQAlpha.Config")
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        # Determine file type by extension
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger = logging.getLogger("NQAlpha.Config")
            logger.warning(f"Unsupported configuration format: {config_path}")
            return {}
        
        return config
        
    except Exception as e:
        logger = logging.getLogger("NQAlpha.Config")
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def apply_resource_limits(args):
    """Apply resource limits
    
    Args:
        args: Command line arguments
    """
    try:
        # Import resource limiter
        import resource
        
        # Apply memory limit
        if args.memory_limit > 0:
            # Convert GB to bytes
            memory_bytes = int(args.memory_limit * 1024 * 1024 * 1024)
            
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            logger = logging.getLogger("NQAlpha.Resources")
            logger.info(f"Memory limit set to {args.memory_limit} GB")
        
        # Apply CPU limit
        if args.cpu_limit > 0:
            # Note: This is an imperfect approach, but it's what we can do in Python
            # For better CPU limiting, use Docker or cgroups
            import psutil
            process = psutil.Process()
            
            # Get available CPU IDs
            all_cpus = list(range(psutil.cpu_count()))
            
            # Limit to specified number of CPUs
            if args.cpu_limit < len(all_cpus):
                cpu_subset = all_cpus[:args.cpu_limit]
                process.cpu_affinity(cpu_subset)
                
                logger = logging.getLogger("NQAlpha.Resources")
                logger.info(f"CPU limit set to {args.cpu_limit} cores")
            
    except Exception as e:
        logger = logging.getLogger("NQAlpha.Resources")
        logger.warning(f"Failed to apply resource limits: {str(e)}")

def handle_signals(controller):
    """Setup signal handlers
    
    Args:
        controller: Master controller instance
    """
    def signal_handler(sig, frame):
        logger = logging.getLogger("NQAlpha.Signal")
        logger.info(f"Received signal {signal.Signals(sig).name}, shutting down...")
        
        # Stop the controller
        controller.stop()
        
        # Exit gracefully
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def interactive_shell(controller):
    """Start interactive shell for system management
    
    Args:
        controller: Master controller instance
    """
    try:
        # Try to import IPython
        from IPython import embed
        
        # Create banner with help info
        banner = """
NQ Alpha Elite Interactive Shell

Available objects:
- controller: Master Controller instance

Example commands:
- controller.get_status()                      # Get system status
- controller.execute_command('restart')        # Restart the system
- controller.get_component('risk_manager')     # Get a component instance
- controller.get_component_status('api_server') # Get component status
        """
        
        # Start IPython shell
        embed(banner1=banner, header='NQ Alpha Elite Interactive Shell')
        
    except ImportError:
        # Fall back to basic interactive Python shell
        import code
        
        # Create local variables
        variables = {
            'controller': controller,
        }
        
        # Print help info
        print("""
NQ Alpha Elite Interactive Shell

Available objects:
- controller: Master Controller instance

Example commands:
- controller.get_status()                      # Get system status
- controller.execute_command('restart')        # Restart the system
- controller.get_component('risk_manager')     # Get a component instance
- controller.get_component_status('api_server') # Get component status
        """)
        
        # Start interactive shell
        code.interact(local=variables)

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = args.log_file or f"logs/nq_elite_{timestamp}.log"
    setup_logging(args.log_level, log_file)
    
    # Get logger
    logger = logging.getLogger("NQAlpha.Main")
    
    # Handle version command
    if args.command == 'version':
        print("NQ Alpha Elite Trading System - Version 5.0.0")
        print(f"Build date: 2025-05-11")
        print(f"Python version: {sys.version}")
        return 0
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.mode:
        if 'system' not in config:
            config['system'] = {}
        config['system']['mode'] = args.mode
    
    if args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(',')]
        if 'trading' not in config:
            config['trading'] = {}
        config['trading']['symbols'] = symbol_list
    
    if args.diagnostics:
        if 'components' not in config:
            config['components'] = {}
        if 'system_diagnostics' not in config['components']:
            config['components']['system_diagnostics'] = {}
        config['components']['system_diagnostics']['enabled'] = True
    
    if args.safe_mode:
        if 'system' not in config:
            config['system'] = {}
        config['system']['safe_mode'] = True
    
    if args.enable_quantum:
        if 'components' not in config:
            config['components'] = {}
        if 'quantum_strategy' not in config['components']:
            config['components']['quantum_strategy'] = {}
        config['components']['quantum_strategy']['enabled'] = True
    
    if args.api_only:
        # Disable all components except API server
        if 'components' not in config:
            config['components'] = {}
        
        for component in config.get('components', {}):
            if component != 'api_server':
                config['components'][component]['enabled'] = False
        
        # Enable API server
        if 'api_server' not in config['components']:
            config['components']['api_server'] = {}
        config['components']['api_server']['enabled'] = True
    
    if args.no_dashboard:
        if 'components' not in config:
            config['components'] = {}
        if 'dashboard' not in config['components']:
            config['components']['dashboard'] = {}
        config['components']['dashboard']['enabled'] = False
    
    # Apply resource limits
    apply_resource_limits(args)
    
    # Initialize master controller with config
    logger.info("Initializing Master Controller")
    controller = MasterController(config)
    
    # Setup signal handlers
    handle_signals(controller)
    
    # Execute command
    if args.command == 'start':
        # Start the system
        logger.info("Starting NQ Alpha Elite")
        
        if not controller.start():
            logger.error("Failed to start NQ Alpha Elite")
            return 1
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            controller.stop()
    
    elif args.command == 'status':
        # Print system status
        status = controller.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'stop':
        # Stop the system
        logger.info("Stopping NQ Alpha Elite")
        controller.stop()
    
    elif args.command == 'restart':
        # Restart the system
        logger.info("Restarting NQ Alpha Elite")
        controller.execute_command('restart')
    
    elif args.command == 'shell':
        # Start interactive shell
        logger.info("Starting interactive shell")
        
        # Start the system if not already running
        if not controller.initialized:
            if not controller.start():
                logger.error("Failed to start NQ Alpha Elite")
                return 1
        
        # Start interactive shell
        interactive_shell(controller)
        
        # Stop the system when shell exits
        controller.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())