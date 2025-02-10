# logger_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Common formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s - [Line: %(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3]
    )
    
    # File handler
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/system_{current_time}.log')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure external package loggers
    for logger_name in ['ids_peak','detectSun']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # Don't add handlers - they'll inherit from root