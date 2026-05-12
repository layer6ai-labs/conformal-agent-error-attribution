import logging
import os
from typing import Optional


def get_logger(name: str = "agentic_conformal", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # If log_file is provided, always add a file handler
    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                            for h in logger.handlers):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(fh)

    # Always ensure console handler exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger

