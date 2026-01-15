"""
Statistical Analysis Module

This module contains all statistical analysis functions and classes
extracted from the original monolithic application.

Note: Renamed from 'statistics' to 'stat_analysis' to avoid conflict 
with Python's built-in statistics module.
"""

from .descriptive_stats import DescriptiveStatistics

__all__ = [
    'DescriptiveStatistics'
]

# Note: InferentialStatistics and CentralLimitTheorem will be added in future updates