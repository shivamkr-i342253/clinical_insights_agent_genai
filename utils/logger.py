"""
Logging utility for the Clinical Insights Agent.
Provides structured, coloured console logging and file logging.
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger.  On first call the root handler is configured.

    Args:
        name:  Module / node name for the logger.
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    global _configured
    if not _configured:
        _configure_root()
        _configured = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _configure_root() -> None:
    """Configure the root logger with a console handler and optional file handler."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColourFormatter())
    root.addHandler(console)

    # File handler – writes to clinical_agent.log
    log_path = Path("clinical_agent.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)


class _ColourFormatter(logging.Formatter):
    """Minimal ANSI colour formatter for console output."""

    COLOURS = {
        logging.DEBUG:    "\033[37m",    # white
        logging.INFO:     "\033[36m",    # cyan
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        return formatter.format(record)
