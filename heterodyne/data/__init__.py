"""
Data loading and processing module for heterodyne analysis.

This module provides utilities for loading experimental XPCS data from various
formats including HDF5 files from APS and APS-U beamlines.

Authors: Heterodyne Analysis Team
"""

from .xpcs_loader import XPCSDataLoader

__all__ = ["XPCSDataLoader"]
