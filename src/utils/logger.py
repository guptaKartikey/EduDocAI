"""Structured logging utility for EduDocAI."""

import logging
import sys
from config.settings import settings


def setup_logger(name: str = "EduDocAI") -> logging.Logger:
    """Configures and returns a logger instance with consistent formatting.

    Args:
        name: Name of the logger module.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Default logger
logger = setup_logger()
