import logging
import os
import sys


logging.basicConfig(
    format=f"%(asctime)s - %(levelname)s - %(name)s - Rank {int(os.environ.get('LOCAL_RANK', '0'))} - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)


BITDISTILLER_DEBUG = os.environ.get("BITDISTILLER_DEBUG", "0") == "1"
FSDP_DEBUG = os.environ.get("FSDP_DEBUG", "0") == "1"

def log_info(logger: logging.Logger, message: str):
    logger.info(message)

def log_fsdp_debug(logger: logging.Logger, message: str):
    if FSDP_DEBUG:
        logger.info(message)

def log_bitdistiller_debug(logger: logging.Logger, message: str):
    if BITDISTILLER_DEBUG:
        logger.info(message)
