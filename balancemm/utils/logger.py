from loguru import logger
import sys

def setup_logger(log_file: str):
    logger.remove()
    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.add(sys.stdout, level="INFO")
    return logger