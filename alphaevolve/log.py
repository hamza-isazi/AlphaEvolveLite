import logging
import sys
from typing import Optional


def init_logger(name: str = "alphaevolve", debug: bool = False) -> logging.Logger:
    """
    Initialize a logger with configurable verbosity.
    
    Parameters
    ----------
    name : str
        Logger name
    debug : bool
        If True, set log level to DEBUG. Otherwise, set to INFO.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: str = "alphaevolve") -> logging.Logger:
    """
    Get an existing logger instance.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
