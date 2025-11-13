"""
Top-level package for the take-home analytics project.
"""

__all__ = ["config", "run_all"]

from .pipeline import ProjectConfig, run_all  # convenience re-export
