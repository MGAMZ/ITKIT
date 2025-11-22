"""
Utility functions for loading, computing, and saving series_meta.json
"""
import json
import os


def get_series_meta_path(folder: str) -> str:
    """Return the full path to series_meta.json in given folder."""
    return os.path.join(folder, 'series_meta.json')


def load_series_meta(folder: str) -> dict[str, dict] | None:
    """Load series_meta.json if exists, else return None."""
    path = get_series_meta_path(folder)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
