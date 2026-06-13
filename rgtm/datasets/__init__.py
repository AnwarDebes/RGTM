"""Synthetic graph datasets for probing long-range / iterative capacity."""
from .synthetic import (adjacent_pair, long_range_match, parity_on_path,
                        within_distance)

__all__ = ["adjacent_pair", "long_range_match", "parity_on_path",
           "within_distance"]
