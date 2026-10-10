"""Logging configuration shared by the scripts and the Streamlit app."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from asl_recognition import config

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Handler names let us detect handlers we already added.
_CONSOLE_HANDLER_NAME = "asl_console"
_FILE_HANDLER_NAME = "asl_file"


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Send log messages to the console and to a rotating log file.

    Safe to call more than once. Streamlit re-runs the script on every
    interaction, and without this check each run would add another handler
    and every message would be printed several times.

    Args:
        level: Minimum level to log (for example logging.INFO).
        log_file: Optional custom log file path. Defaults to config.LOG_FILE.
    """
    log_path = Path(log_file) if log_file is not None else config.LOG_FILE
    formatter = logging.Formatter(LOG_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    existing_names = {handler.get_name() for handler in root_logger.handlers}

    if _CONSOLE_HANDLER_NAME not in existing_names:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.set_name(_CONSOLE_HANDLER_NAME)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if _FILE_HANDLER_NAME not in existing_names:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Rotating file: keeps the log from growing forever.
            file_handler = RotatingFileHandler(
                log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
        except OSError as error:
            # A missing log file should not stop the app, so only warn.
            root_logger.warning("Could not open log file %s: %s", log_path, error)
        else:
            file_handler.set_name(_FILE_HANDLER_NAME)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)