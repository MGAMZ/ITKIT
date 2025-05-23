"""
Utility functions for loading, computing, and saving series_meta.json
"""
import os
import json
import SimpleITK as sitk


def get_series_meta_path(folder: str) -> str:
    """Return the full path to series_meta.json in given folder."""
    return os.path.join(folder, 'series_meta.json')


def load_series_meta(folder: str) -> list[dict] | None:
    """Load series_meta.json if exists, else return None."""
    path = get_series_meta_path(folder)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def compute_series_meta(folder: str) -> list[dict]:
    """Compute series meta by reading images under folder/image."""
    image_folder = os.path.join(folder, 'image')
    entries: list[dict] = []
    if not os.path.isdir(image_folder):
        return entries
    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(('.mha', '.nii', '.nii.gz', '.mhd')):
            continue
        path = os.path.join(image_folder, fname)
        try:
            img = sitk.ReadImage(path)
            size = list(img.GetSize()[::-1])
            spacing = list(img.GetSpacing()[::-1])
            entries.append({'name': fname, 'size': size, 'spacing': spacing})
        except Exception:
            continue
    return entries


def save_series_meta(series_meta: list[dict], folder: str) -> None:
    """Save series_meta.json under given folder."""
    path = get_series_meta_path(folder)
    with open(path, 'w') as f:
        json.dump(series_meta, f, indent=4)
