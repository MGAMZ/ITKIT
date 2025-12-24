"""
Utility functions for loading, computing, and saving meta.json
"""
import json
import os


def get_series_meta_path(folder: str) -> str:
    """Return the full path to meta.json in given folder."""
    return os.path.join(folder, 'meta.json')


def load_series_meta(folder: str) -> dict[str, dict] | None:
    """Load meta.json if exists, else return None."""
    path = get_series_meta_path(folder)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
