import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    """
    Set up a centralized logger for the application.
    This function configures the root logger and then iterates through all
    existing loggers to ensure they are properly configured to propagate
    their logs to the root logger's handlers. This is important for
    capturing logs from third-party libraries like tensorrt-llm that may
    initialize their loggers before this function is called.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = "logs"
    log_file = os.path.join(log_dir, "app.log")

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a shared formatter
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create shared handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, 5 backups
    )
    file_handler.setFormatter(log_format)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Re-configure all existing loggers
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()
            logger.setLevel(log_level)
            logger.propagate = True 