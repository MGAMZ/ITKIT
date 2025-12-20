import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def get_meta_info(itk_image:sitk.Image):
    return {
        'origin': itk_image.GetOrigin()[::-1],
        'spacing': itk_image.GetSpacing()[::-1],
        'size': itk_image.GetSize()[::-1],
    }


def get_SeriesUID_from_filename(filename:str):
    return Path(filename).stem


def calculate_quantitative_metrics(img_itk:sitk.Image,
                                   gt_itk:sitk.Image,
                                   pred_itk:sitk.Image,
                                   num_classes:int):
    """ Calculate Dice, IoU, Accuracy, Recall, Precision using scikit-learn

    Args:
        img_itk: Original image (not used in metrics but kept for future extensions)
        gt_itk: Ground truth segmentation mask
        pred_itk: Predicted segmentation mask
        num_classes: Number of classes including background

    Returns:
        dict: Metrics for each class, keys are metric names with class indices
    """
    from sklearn.metrics import (
        accuracy_score,
        jaccard_score,
        precision_recall_fscore_support,
    )

    # Convert SimpleITK images to numpy arrays
    gt_array = sitk.GetArrayFromImage(gt_itk).flatten()
    pred_array = sitk.GetArrayFromImage(pred_itk).flatten()

    # Overall accuracy
    overall_accuracy = accuracy_score(gt_array, pred_array)

    # Per-class metrics using scikit-learn
    # Set zero_division=0 to handle empty classes gracefully
    precision, recall, f1score, support = precision_recall_fscore_support(
        gt_array,
        pred_array,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )

    # IoU (Jaccard) per class
    iou = jaccard_score(
        gt_array,
        pred_array,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )

    # Convert to numpy arrays to ensure proper type (avoiding type checker warnings)
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    f1score = np.asarray(f1score)
    support = np.asarray(support)
    iou = np.asarray(iou)

    # Dice coefficient equals F1 score for binary classification per class
    dice = f1score

    # Organize results
    results = {
        'overall_accuracy': overall_accuracy,
    }

    for class_idx in range(num_classes):
        results[f'class_{class_idx}_dice'] = dice[class_idx]
        results[f'class_{class_idx}_iou'] = iou[class_idx]
        results[f'class_{class_idx}_precision'] = precision[class_idx]
        results[f'class_{class_idx}_recall'] = recall[class_idx]
        results[f'class_{class_idx}_support'] = support[class_idx]

    return results


class Evaluator:
    def __init__(self, num_classes:int, workers:int|None=None, class_names:list[str]|None=None):
        self.num_classes = num_classes
        self.workers = workers or mp.cpu_count()
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]

    def executor(self, img_dir:str, gt_dir:str, pred_dir:str, save_path:str):
        """ Execute evaluation on multiple samples with multiprocessing

        Args:
            img_dir: Directory containing original images (.mha files)
            gt_dir: Directory containing ground truth masks (.mha files)
            pred_dir: Directory containing prediction masks (.mha files)
            save_path: Path to save the evaluation results (xlsx file)
        """
        # Get all .mha files from each directory
        img_dir_path = Path(img_dir)
        gt_dir_path = Path(gt_dir)
        pred_dir_path = Path(pred_dir)

        # Find all .mha files and extract their stems (filenames without extension)
        img_files = {f.stem: f for f in img_dir_path.glob('*.mha')}
        gt_files = {f.stem: f for f in gt_dir_path.glob('*.mha')}
        pred_files = {f.stem: f for f in pred_dir_path.glob('*.mha')}

        # Get intersection of all three sets (common SeriesUIDs)
        common_series_uids = set(img_files.keys()) & set(gt_files.keys()) & set(pred_files.keys())

        if not common_series_uids:
            raise ValueError("No common .mha files found in all three directories")

        print(f"Found {len(common_series_uids)} common samples across all directories")
        print(f"  Images directory: {len(img_files)} .mha files")
        print(f"  Ground truth directory: {len(gt_files)} .mha files")
        print(f"  Predictions directory: {len(pred_files)} .mha files")

        # Sort for consistent ordering
        common_series_uids = sorted(common_series_uids)

        # Build task list with matched file paths
        tasks = [
            (str(img_files[uid]), str(gt_files[uid]), str(pred_files[uid]))
            for uid in common_series_uids
        ]

        results_list = []

        with mp.Pool(processes=self.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(self._calculate_one_sample, tasks),
                total=len(tasks),
                desc="Evaluating samples"
            ):
                results_list.append(result)

        self._save_results(results_list, save_path)

        return results_list

    def _calculate_one_sample(self, args:tuple[str, str, str]):
        """ Calculate metrics for one sample

        Args:
            img_path: Path to original image
            gt_path: Path to ground truth mask
            pred_path: Path to prediction mask

        Returns:
            dict: Metrics with SeriesUID and all evaluation metrics
        """
        img_path, gt_path, pred_path = args

        img_itk = sitk.ReadImage(img_path)
        gt_itk = sitk.ReadImage(gt_path)
        pred_itk = sitk.ReadImage(pred_path)

        series_uid = get_SeriesUID_from_filename(pred_path)
        quantitative_metrics = calculate_quantitative_metrics(img_itk, gt_itk, pred_itk, self.num_classes)

        # Add SeriesUID and metadata
        result = {
            'SeriesUID': series_uid,
            **get_meta_info(pred_itk),
            **quantitative_metrics
        }

        return result

    def _save_results(self, results_list:list[dict], save_path:str):
        """ Convert results to xlsx table with organized columns

        Args:
            results_list: List of result dictionaries from each sample
            save_path: Path to save the xlsx file
        """
        if not results_list:
            raise FileNotFoundError("No results to save")

        # Convert to DataFrame
        df = pd.DataFrame(results_list)

        # Organize columns: SeriesUID first, then metadata, then metrics grouped by class
        base_columns = ['SeriesUID', 'origin', 'spacing', 'size', 'overall_accuracy']

        # Group metrics by class
        metric_types = ['dice', 'iou', 'precision', 'recall', 'support']
        class_columns = []

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            for metric_type in metric_types:
                col_key = f'class_{class_idx}_{metric_type}'
                if col_key in df.columns:
                    # Rename column to include class name for readability
                    new_col_name = f'{class_name}_{metric_type.capitalize()}'
                    df.rename(columns={col_key: new_col_name}, inplace=True)
                    class_columns.append(new_col_name)

        # Reorder columns
        ordered_columns = base_columns + class_columns
        ordered_columns = [col for col in ordered_columns if col in df.columns]
        df = df[ordered_columns]

        # Save and Print
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(save_path, index=False)
        print(f"Results saved to: {save_path}")
        print(f"Total samples evaluated: {len(results_list)}")


def parse_arg():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results.")
    parser.add_argument('-i', '--img-dir', type=str)
    parser.add_argument('-g', '--gt-dir', type=str)
    parser.add_argument('-p', '--pred-dir', type=str)
    parser.add_argument('-s', '--save-path', type=str, default='evaluation_results.xlsx')
    parser.add_argument('-n', '--num-classes', type=int)
    parser.add_argument('-c', '--class-names', type=str, nargs='*', default=None)
    parser.add_argument('-w', '--workers', type=int, default=None)
    return parser.parse_args()


# Example usage
if __name__ == '__main__':
    args = parse_arg()

    # Example: Evaluate a set of predictions
    evaluator = Evaluator(
        num_classes=args.num_classes,
        workers=args.workers,
        class_names=args.class_names
    )

    # Execute evaluation by specifying three directories
    evaluator.executor(
        img_dir=args.img_dir,
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        save_path=args.save_path
    )
