# logger_config.py
import logging
import os
from datetime import datetime

class SingleDecimalFormatter(logging.Formatter):
    """Custom formatter that formats datetime with a single decimal place for seconds."""
    def formatTime(self, record, datefmt=None):
        created_time = datetime.fromtimestamp(record.created)
        if datefmt:
            # Format with full microseconds
            s = created_time.strftime(datefmt)
            # Extract only the first decimal place
            if '.' in s:
                parts = s.split('.')
                if len(parts) == 2 and len(parts[1]) >= 1:
                    return f"{parts[0]}.{parts[1][0]}"
        return created_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:21]  # Default with single decimal

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Use custom formatter with 0.1 second precision
    formatter = SingleDecimalFormatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s - [Line: %(lineno)d]',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
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