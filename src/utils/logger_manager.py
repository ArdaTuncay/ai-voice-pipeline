import sys
from pathlib import Path

from loguru import logger

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    format=CONSOLE_FORMAT,
    colorize=True,
    backtrace=False,
    diagnose=False,
)

logger.add(
    LOG_DIR / "pipeline.log",
    level="DEBUG",
    format=FILE_FORMAT,
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    backtrace=True,
    diagnose=True,
)

__all__ = ["logger"]
