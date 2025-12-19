"""
Evaluate segmentation metrics for predictions against ground truth.

This module provides a CLI tool to evaluate segmentation predictions by calculating
various metrics (Dice, IoU, F-score, Recall, Precision, Accuracy) and saving them
in multiple aggregation formats (per-class sample-averaged, per-sample per-class,
per-sample class-averaged).

The tool automatically resamples predictions to match ground truth spacing/size
if they differ, and orients all samples to LPI for consistent evaluation.
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import (
    accuracy_score,
    jaccard_score,
    precision_recall_fscore_support,
)

from itkit.io.sitk_toolkit import sitk_resample_to_image
from itkit.process.base_processor import SeparateFoldersProcessor


def calculate_metrics_for_sample(
    gt_itk: sitk.Image,
    pred_itk: sitk.Image,
    sample_name: str,
) -> dict[str, float | str]:
    """Calculate segmentation metrics for one sample.
    
    Args:
        gt_itk: Ground truth segmentation mask
        pred_itk: Predicted segmentation mask (must match gt_itk in size/spacing)
        sample_name: Name of the sample
        
    Returns:
        Dictionary with metrics for each class and overall accuracy
    """
    # Convert to numpy arrays
    gt_array = sitk.GetArrayFromImage(gt_itk).flatten()
    pred_array = sitk.GetArrayFromImage(pred_itk).flatten()
    
    # Get unique classes from both GT and prediction
    all_classes = np.union1d(np.unique(gt_array), np.unique(pred_array))
    num_classes = len(all_classes)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(gt_array, pred_array)
    
    # Per-class metrics using scikit-learn (authoritative implementation)
    # F1-score equals Dice coefficient for binary classification per class
    precision, recall, f1score, support = precision_recall_fscore_support(
        gt_array,
        pred_array,
        labels=all_classes.tolist(),
        average=None,
        zero_division=0
    )
    
    # IoU (Jaccard) per class
    iou = jaccard_score(
        gt_array,
        pred_array,
        labels=all_classes.tolist(),
        average=None,
        zero_division=0
    )
    
    # Organize results
    result = {'sample': sample_name, 'accuracy': overall_accuracy}
    
    for i, class_idx in enumerate(all_classes):
        result[f'class_{int(class_idx)}_dice'] = f1score[i]
        result[f'class_{int(class_idx)}_iou'] = iou[i]
        result[f'class_{int(class_idx)}_fscore'] = f1score[i]  # F-score = Dice
        result[f'class_{int(class_idx)}_recall'] = recall[i]
        result[f'class_{int(class_idx)}_precision'] = precision[i]
    
    return result


class EvaluateProcessor(SeparateFoldersProcessor):
    """Processor for evaluating segmentation predictions against ground truth.
    
    This processor:
    1. Matches GT and prediction files by filename
    2. Resamples predictions to GT if they have different spacing/size
    3. Orients both to LPI
    4. Calculates metrics for each sample
    """
    
    def __init__(self, gt_folder: str, pred_folder: str, *args, **kwargs):
        """Initialize the evaluation processor.
        
        Args:
            gt_folder: Folder containing ground truth masks
            pred_folder: Folder containing prediction masks
        """
        super().__init__(
            folder_A=gt_folder,
            folder_B=pred_folder,
            *args,
            **kwargs
        )
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.results = []
    
    def process_one(self, args: tuple[str, str]) -> None:
        """Process one GT-prediction pair.
        
        Args:
            args: Tuple of (gt_path, pred_path)
            
        Returns:
            None (results are stored in self.results)
        """
        gt_path, pred_path = args
        sample_name = Path(gt_path).stem
        
        # Read images
        gt_itk = sitk.ReadImage(gt_path)
        pred_itk = sitk.ReadImage(pred_path)
        
        # Orient both to LPI for consistent evaluation
        gt_itk = sitk.DICOMOrient(gt_itk, 'LPI')
        pred_itk = sitk.DICOMOrient(pred_itk, 'LPI')
        
        # Check if resampling is needed
        needs_resampling = (
            gt_itk.GetSize() != pred_itk.GetSize() or
            gt_itk.GetSpacing() != pred_itk.GetSpacing()
        )
        
        if needs_resampling:
            print(f"Resampling {sample_name}: pred shape {pred_itk.GetSize()} -> gt shape {gt_itk.GetSize()}")
            pred_itk = sitk_resample_to_image(
                image=pred_itk,
                reference_image=gt_itk,
                field='label',
                default_value=0.0
            )
        
        # Calculate metrics
        result = calculate_metrics_for_sample(gt_itk, pred_itk, sample_name)
        self.results.append(result)
        
        return None


def aggregate_metrics(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate metrics into three different views.
    
    Args:
        results: List of per-sample metric dictionaries
        
    Returns:
        Tuple of (per_class_sample_avg, per_sample_per_class, per_sample_class_avg)
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Extract metric columns (exclude 'sample' and 'accuracy')
    metric_cols = [col for col in df.columns if col not in ['sample', 'accuracy']]
    
    # Get unique classes from column names
    classes = set()
    for col in metric_cols:
        if col.startswith('class_'):
            class_idx = int(col.split('_')[1])
            classes.add(class_idx)
    classes = sorted(classes)
    
    # 1. Per-sample per-class metrics (most detailed)
    # Columns: sample, class_0_dice, class_0_iou, ..., class_N_precision, accuracy
    per_sample_per_class = df.copy()
    
    # 2. Per-class sample-averaged metrics
    # For each class and metric, compute mean across all samples
    per_class_data = {'metric': []}
    for class_idx in classes:
        per_class_data[f'class_{class_idx}'] = []
    
    metrics_to_aggregate = ['dice', 'iou', 'fscore', 'recall', 'precision']
    for metric in metrics_to_aggregate:
        per_class_data['metric'].append(metric)
        for class_idx in classes:
            col_name = f'class_{class_idx}_{metric}'
            if col_name in df.columns:
                per_class_data[f'class_{class_idx}'].append(df[col_name].mean())
            else:
                per_class_data[f'class_{class_idx}'].append(np.nan)
    
    # Add accuracy row
    per_class_data['metric'].append('accuracy')
    for class_idx in classes:
        per_class_data[f'class_{class_idx}'].append(df['accuracy'].mean())
    
    per_class_sample_avg = pd.DataFrame(per_class_data)
    
    # 3. Per-sample class-averaged metrics
    # For each sample, compute mean across all classes for each metric
    per_sample_class_avg_data = {'sample': df['sample'].tolist()}
    
    for metric in metrics_to_aggregate:
        metric_values = []
        for _, row in df.iterrows():
            class_values = []
            for class_idx in classes:
                col_name = f'class_{class_idx}_{metric}'
                if col_name in row:
                    class_values.append(row[col_name])
            # Average across classes
            if class_values:
                metric_values.append(np.mean(class_values))
            else:
                metric_values.append(np.nan)
        per_sample_class_avg_data[metric] = metric_values
    
    # Add accuracy (same as per-sample)
    per_sample_class_avg_data['accuracy'] = df['accuracy'].tolist()
    
    per_sample_class_avg = pd.DataFrame(per_sample_class_avg_data)
    
    return per_class_sample_avg, per_sample_per_class, per_sample_class_avg


def save_results(
    per_class_sample_avg: pd.DataFrame,
    per_sample_per_class: pd.DataFrame,
    per_sample_class_avg: pd.DataFrame,
    save_folder: str,
    format: Literal['csv', 'excel']
):
    """Save the three metric tables to files.
    
    Args:
        per_class_sample_avg: Per-class sample-averaged metrics
        per_sample_per_class: Per-sample per-class metrics
        per_sample_class_avg: Per-sample class-averaged metrics
        save_folder: Directory to save results
        format: Output format ('csv' or 'excel')
    """
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    if format == 'csv':
        # Save as three separate CSV files
        per_class_sample_avg.to_csv(
            os.path.join(save_folder, 'per_class_sample_avg.csv'),
            index=False
        )
        per_sample_per_class.to_csv(
            os.path.join(save_folder, 'per_sample_per_class.csv'),
            index=False
        )
        per_sample_class_avg.to_csv(
            os.path.join(save_folder, 'per_sample_class_avg.csv'),
            index=False
        )
        print(f"Results saved to {save_folder}/*.csv")
        
    elif format == 'excel':
        # Save as one Excel file with three sheets
        excel_path = os.path.join(save_folder, 'evaluation_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            per_class_sample_avg.to_excel(writer, sheet_name='PerClass_SampleAvg', index=False)
            per_sample_per_class.to_excel(writer, sheet_name='PerSample_PerClass', index=False)
            per_sample_class_avg.to_excel(writer, sheet_name='PerSample_ClassAvg', index=False)
        print(f"Results saved to {excel_path}")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'excel'.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions against ground truth. "
                    "Calculates Dice, IoU, F-score, Recall, Precision, and Accuracy metrics."
    )
    parser.add_argument(
        'gt_folder',
        type=str,
        help='Folder containing ground truth masks (.mha, .nii, .nii.gz, .nrrd files)'
    )
    parser.add_argument(
        'pred_folder',
        type=str,
        help='Folder containing prediction masks (same format as gt_folder)'
    )
    parser.add_argument(
        'save_folder',
        type=str,
        help='Folder to save evaluation results (created if not exists)'
    )
    parser.add_argument(
        'format',
        type=str,
        choices=['csv', 'excel'],
        help='Output format: "csv" (3 files) or "excel" (1 file with 3 sheets)'
    )
    parser.add_argument(
        '--mp',
        action='store_true',
        help='Enable multiprocessing for faster evaluation'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: half of CPU cores)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the itk_evaluate CLI command."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.gt_folder):
        raise ValueError(f"Ground truth folder does not exist: {args.gt_folder}")
    if not os.path.isdir(args.pred_folder):
        raise ValueError(f"Prediction folder does not exist: {args.pred_folder}")
    
    print(f"Evaluating predictions...")
    print(f"  Ground truth folder: {args.gt_folder}")
    print(f"  Prediction folder: {args.pred_folder}")
    print(f"  Save folder: {args.save_folder}")
    print(f"  Output format: {args.format}")
    
    # Create processor and run evaluation
    processor = EvaluateProcessor(
        gt_folder=args.gt_folder,
        pred_folder=args.pred_folder,
        mp=args.mp,
        workers=args.workers
    )
    
    # Process all samples
    processor.process("Evaluating segmentation")
    
    if not processor.results:
        print("No matching samples found for evaluation.")
        return
    
    print(f"Evaluated {len(processor.results)} samples.")
    
    # Aggregate metrics into three views
    per_class_sample_avg, per_sample_per_class, per_sample_class_avg = aggregate_metrics(
        processor.results
    )
    
    # Save results
    save_results(
        per_class_sample_avg,
        per_sample_per_class,
        per_sample_class_avg,
        args.save_folder,
        args.format
    )
    
    # Print summary
    print("\nEvaluation complete!")
    print("\nThree types of metric tables saved:")
    print("1. Per-class sample-averaged: Mean metric for each class across all samples")
    print("2. Per-sample per-class: Detailed metrics for each sample and class")
    print("3. Per-sample class-averaged: Mean metric across classes for each sample")


if __name__ == '__main__':
    main()
