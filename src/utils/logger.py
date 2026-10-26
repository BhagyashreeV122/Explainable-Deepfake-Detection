"""Logging utilities for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "deepfake_detector", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a structured logger with both console and optional file handlers.

    Args:
        name: Name of the logger.
        log_file: Optional path to save log messages to disk.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
