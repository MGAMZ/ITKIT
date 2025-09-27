import os
import json
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union
import SimpleITK as sitk


class ProcessingMode(Enum):
    """Define different file processing modes"""
    DATASET_PAIRS = "dataset_pairs"      # image/label subfolders
    SINGLE_FOLDER = "single_folder"      # single folder with files
    SEPARATE_FOLDERS = "separate_folders"  # separate img and lbl folders
    

class BaseITKProcessor(ABC):
    """Base class for ITK processing with flexible file discovery"""
    
    def __init__(self, source_folder: str = None, dest_folder: str = None, 
                 mp: bool = False, workers: Optional[int] = None, 
                 extensions: Tuple[str, ...] = ('.mha', '.mhd', '.nii', '.nii.gz')):
        self.source_folder = source_folder
        self.dest_folder = dest_folder
        self.mp = mp
        self.workers = workers or cpu_count()
        self.meta = {}
        self.extensions = extensions
    
    def find_files_recursive(self, folder: str, extensions: Tuple[str, ...] = None) -> List[str]:
        """Recursively find files with given extensions"""
        if extensions is None:
            extensions = self.extensions
            
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(extensions):
                    files.append(os.path.join(root, filename))
        return files
    
    def find_files_flat(self, folder: str, extensions: Tuple[str, ...] = None) -> List[str]:
        """Find files in folder (non-recursive)"""
        if extensions is None:
            extensions = self.extensions
            
        files = []
        for f in os.listdir(folder):
            if f.endswith(extensions):
                files.append(os.path.join(folder, f))
        return files
    
    def find_pairs_dataset(self, source_folder: str, recursive: bool = False) -> List[Tuple[str, str]]:
        """Find image-label pairs in dataset structure (image/ and label/ subfolders)"""
        img_dir = os.path.join(source_folder, 'image')
        lbl_dir = os.path.join(source_folder, 'label')
        
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            raise ValueError(f"Missing 'image' or 'label' subfolders in {source_folder}")
        
        if recursive:
            # Get all image files with relative paths
            img_files = {}
            for img_path in self.find_files_recursive(img_dir):
                rel_path = os.path.relpath(img_path, img_dir)
                # Normalize extension for matching
                key = self._normalize_filename(rel_path)
                img_files[key] = img_path
            
            lbl_files = {}
            for lbl_path in self.find_files_recursive(lbl_dir):
                rel_path = os.path.relpath(lbl_path, lbl_dir)
                key = self._normalize_filename(rel_path)
                lbl_files[key] = lbl_path
        else:
            # Simple flat matching
            img_files = {f: os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(self.extensions)}
            lbl_files = {f: os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(self.extensions)}
        
        # Find intersection
        common_keys = set(img_files.keys()) & set(lbl_files.keys())
        pairs = [(img_files[key], lbl_files[key]) for key in common_keys]
        return pairs
    
    def find_pairs_separate(self, img_folder: str, lbl_folder: str) -> List[Tuple[str, str]]:
        """Find image-label pairs in separate folders"""
        img_files = {os.path.basename(f): f for f in self.find_files_flat(img_folder)}
        lbl_files = {os.path.basename(f): f for f in self.find_files_flat(lbl_folder)}
        
        common_files = set(img_files.keys()) & set(lbl_files.keys())
        pairs = [(img_files[f], lbl_files[f]) for f in common_files]
        return pairs
    
    def _normalize_filename(self, filepath: str) -> str:
        """Normalize filename for matching (remove extension differences)"""
        base = os.path.splitext(filepath)[0]
        # Handle .nii.gz
        if base.endswith('.nii'):
            base = base[:-4]
        return base
    
    def process_items(self, items: List, desc: str = "Processing"):
        """Generic processing with multiprocessing support"""
        if not items:
            print(f"No items found for {desc.lower()}.")
            return []
        
        if self.mp:
            with Pool(self.workers) as pool:
                results = list(tqdm(pool.imap_unordered(self.process_one, items),
                                    total=len(items), desc=desc, dynamic_ncols=True))
        else:
            results = []
            for item in tqdm(items, desc=desc, dynamic_ncols=True):
                results.append(self.process_one(item))
        
        # Collect metadata
        for res in results:
            if res:
                self.meta.update(res)
        
        return results
    
    @abstractmethod
    def process_one(self, args) -> Optional[Dict]:
        """Process one item - must be implemented by subclasses"""
        pass
    
    def save_meta(self, meta_path: str = None):
        """Save metadata to JSON"""
        if not meta_path and self.dest_folder:
            meta_path = os.path.join(self.dest_folder, 'series_meta.json')
        if meta_path:
            try:
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, 'w') as f:
                    json.dump(self.meta, f, indent=4)
                print(f"Metadata saved to {meta_path}")
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")


class DatasetProcessor(BaseITKProcessor):
    """For processing datasets with image/label structure"""
    
    def __init__(self, source_folder: str, dest_folder: str = None, 
                 mp: bool = False, workers: Optional[int] = None, 
                 recursive: bool = False):
        super().__init__(source_folder, dest_folder, mp, workers)
        self.recursive = recursive
    
    def find_pairs(self) -> List[Tuple[str, str]]:
        """Find image-label pairs"""
        return self.find_pairs_dataset(self.source_folder, self.recursive)
    
    def process(self):
        """Process all pairs"""
        pairs = self.find_pairs()
        self.process_items(pairs, "Processing pairs")


class SingleFolderProcessor(BaseITKProcessor):
    """For processing single folder of files"""
    
    def __init__(self, source_folder: str, dest_folder: str = None, 
                 mp: bool = False, workers: Optional[int] = None, 
                 recursive: bool = False):
        super().__init__(source_folder, dest_folder, mp, workers)
        self.recursive = recursive
    
    def find_files(self) -> List[str]:
        """Find files to process"""
        if self.recursive:
            return self.find_files_recursive(self.source_folder)
        else:
            return self.find_files_flat(self.source_folder)
    
    def process(self):
        """Process all files"""
        files = self.find_files()
        self.process_items(files, "Processing files")


class SeparateFoldersProcessor(BaseITKProcessor):
    """For processing separate image and label folders"""
    
    def __init__(self, img_folder: str, lbl_folder: str, 
                 out_img_folder: str = None, out_lbl_folder: str = None,
                 mp: bool = False, workers: Optional[int] = None):
        super().__init__(None, None, mp, workers)
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.out_img_folder = out_img_folder
        self.out_lbl_folder = out_lbl_folder
    
    def find_pairs(self) -> List[Tuple[str, str]]:
        """Find image-label pairs from separate folders"""
        return self.find_pairs_separate(self.img_folder, self.lbl_folder)
    
    def process(self):
        """Process all pairs"""
        pairs = self.find_pairs()
        self.process_items(pairs, "Processing pairs")