import logging
import os
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "huracan") -> logging.Logger:
    """
    Get a pre-configured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger is not configured, set it up with default settings
    if not logger.handlers:
        # Check for log directory in project structure
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{name}.log"
        
        return setup_logger(
            name=name,
            log_level=logging.INFO,
            log_file=str(log_file),
            console_output=True
        )
    
    return logger 