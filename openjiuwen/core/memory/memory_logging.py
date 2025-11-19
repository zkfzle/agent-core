import logging
import  sys
from typing import Optional

# Global Log Object
_global_logger = logging.getLogger("memory_logger")
# Avoid Repeated Addition of Handlers
if not _global_logger.handlers:
    # Set Default Log Level
    _global_logger.setLevel(logging.INFO)

    # Configure Console Output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Configure Log Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H-%M-%S'
    )
    console_handler.setFormatter(formatter)

    # Add Handlers

    _global_logger.addHandler(console_handler)


def set_logger(custom_logger: Optional[logging.Logger] = None):
    """
    Set Global Log Object
    Parameters:
        custom_logger: Custom log object; if provided, this object will be used
    """
    global _global_logger

    # If a custom log object is provided, use it and update the global object
    if custom_logger is not None:
        _global_logger = custom_logger

def get_logger():
    """Obtain the global logger object"""
    return _global_logger