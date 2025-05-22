"""
Central logging configuration for the project.

This module provides centralized logging configuration and utilities
for consistent logging across all modules.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

def configure_logging(config: Dict[str, Any], debug: bool = False) -> None:
    """Configure logging for the entire project.
    
    Args:
        config: Configuration dictionary
        debug: Whether to enable debug logging
    """
    # Get log configuration
    log_config = config.get('logging', {})
    
    # Set base log level
    base_level = logging.DEBUG if debug else logging.INFO
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(base_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(base_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log directory specified
    log_dir = log_config.get('log_dir')
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / 'medproj.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(base_level)
        root_logger.addHandler(file_handler)
    
    # Configure module-specific log levels
    module_levels = log_config.get('module_levels', {})
    for module, level in module_levels.items():
        module_logger = logging.getLogger(module)
        module_logger.setLevel(getattr(logging, level.upper()))
    
    # Log configuration
    logging.info("Logging configured with:")
    logging.info(f"  Base level: {logging.getLevelName(base_level)}")
    logging.info(f"  Debug mode: {debug}")
    if log_dir:
        logging.info(f"  Log directory: {log_dir}")
    if module_levels:
        logging.info("  Module-specific levels:")
        for module, level in module_levels.items():
            logging.info(f"    {module}: {level}")

def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """Get a logger with the specified name and debug level.
    
    Args:
        name: Logger name (typically __name__)
        debug: Whether to enable debug logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if debug:
        logger.setLevel(logging.DEBUG)
    return logger

def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """Log configuration details.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convert config to JSON for pretty printing
    config_json = json.dumps(config, indent=2)
    
    logger.info("Configuration:")
    logger.info(f"\n{config_json}")

def log_error(error: Exception, logger: Optional[logging.Logger] = None,
             include_traceback: bool = True) -> None:
    """Log an error with optional traceback.
    
    Args:
        error: Exception to log
        logger: Optional logger instance
        include_traceback: Whether to include traceback
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if include_traceback:
        logger.exception(f"Error: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")

def log_warning(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a warning message.
    
    Args:
        message: Warning message
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.warning(message)

def log_info(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log an info message.
    
    Args:
        message: Info message
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(message)

def log_debug(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a debug message.
    
    Args:
        message: Debug message
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.debug(message) 