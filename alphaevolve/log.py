import logging
import sys


def init_logger(name: str = "alphaevolve") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger
