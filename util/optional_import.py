"""
module for optional imports
- inspired by how pandas handles optional imports
"""
import importlib
import warnings


def import_optional(name, behavior = "warn"):
    """
    Import a module and handle the case where it is not installed.
    Parameters:
        name (str): The name of the module to import.
        behavior (str): The behavior to take if the module is not found.
                        Options are "warn", "raise", or "ignore".
    Returns:
        module: The imported module, or None if it was not found and behavior is "ignore", or "warn".
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        if behavior == "raise":
            raise ImportError(f"Module '{name}' not found.")
        elif behavior == "warn":
            warnings.warn(f"Module '{name}' not found. Some functionality, such as automatic table computations, may be limited.")
            module = None
        elif behavior == "ignore":
            module = None
        else:
            raise ValueError(f"Invalid behavior '{behavior}'. Choose from 'warn', 'raise', or 'ignore'.")
    return module
