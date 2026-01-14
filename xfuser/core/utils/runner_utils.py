import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s - %(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

def log(message: str, debug=False) -> None:
    """Log message only from the last process to avoid duplicates."""
    if is_last_process():
        if debug:
            logger.debug(message)
        else:
            logger.info(message)

def is_last_process() -> bool:
    """ Checks based on env rank and world size if this is last process in """
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    if rank == world_size - 1:
        return True
    return False