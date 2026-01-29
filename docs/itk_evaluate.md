# itk_evaluate

Evaluate segmentation predictions against ground truth by calculating comprehensive metrics and volume statistics.

## Usage

```bash
itk_evaluate <gt_folder> <pred_folder> <save_folder> [options]
```

## Parameters

- `gt_folder`: Folder containing ground truth masks (.mha, .nii, .nii.gz, .nrrd files)
- `pred_folder`: Folder containing prediction masks (same format as gt_folder)
- `save_folder`: Folder to save evaluation results (created if not exists)
- `--format {csv,excel}`: Output format (default: excel)
  - `csv`: Saves 5 separate CSV files
  - `excel`: Saves 1 Excel file with 5 sheets
- `--mp`: Enable multiprocessing for faster evaluation
- `--workers N`: Number of worker processes (default: half of CPU cores)

## Features

- **Automatic Resampling**: Predictions are automatically resampled to match ground truth spacing/size if they differ
- **Consistent Orientation**: All samples are oriented to LPI for consistent evaluation
- **Multi-class Support**: Handles multi-class segmentation with per-class metrics
- **Volume Statistics**: Calculates volumes in cubic millimeters for both GT and predictions
- **Multiple Aggregation Views**: Provides metrics in 5 different aggregation formats

## Metrics Calculated

For each class, the following metrics are computed:

- **Dice Coefficient**: Measures overlap between GT and prediction (F1-score)
- **IoU (Jaccard)**: Intersection over Union
- **F-score**: Same as Dice for segmentation tasks
- **Recall (Sensitivity)**: True positive rate
- **Precision**: Positive predictive value
- **Accuracy**: Overall pixel-wise accuracy (global metric)

## Volume Statistics

For each class in each sample:

- **Volume_GT**: Volume in cubic millimeters calculated from ground truth
- **Volume_Pred**: Volume in cubic millimeters calculated from prediction

Volumes are computed as: `voxel_count × (spacing_x × spacing_y × spacing_z)`

## Output Tables

The tool generates 5 different views of the evaluation results:

### 1. Per-Class Sample-Averaged (`PerClass_SampleAvg`)

Mean metrics for each class across all samples.

| metric | class_0 | class_1 | class_2 | ... |
|--------|---------|---------|---------|-----|
| dice   | 0.95    | 0.92    | 0.88    | ... |
| iou    | 0.91    | 0.85    | 0.79    | ... |
| recall | 0.94    | 0.91    | 0.87    | ... |

### 2. Per-Sample Per-Class (`PerSample_PerClass`)

Detailed metrics for each sample and each class (most granular view).

| sample  | class_0_dice | class_0_iou | class_1_dice | ... | accuracy |
|---------|--------------|-------------|--------------|-----|----------|
| case001 | 0.96         | 0.92        | 0.94         | ... | 0.95     |
| case002 | 0.94         | 0.89        | 0.90         | ... | 0.93     |

### 3. Per-Sample Class-Averaged (`PerSample_ClassAvg`)

Mean metric across all classes for each sample.

| sample  | dice | iou  | fscore | recall | precision | accuracy |
|---------|------|------|--------|--------|-----------|----------|
| case001 | 0.95 | 0.91 | 0.95   | 0.94   | 0.96      | 0.95     |
| case002 | 0.92 | 0.85 | 0.92   | 0.91   | 0.93      | 0.93     |

### 4. Volume_GT

Volume in cubic millimeters for each class in each sample (from ground truth).

| sample  | class_0 | class_1 | class_2 | ... |
|---------|---------|---------|---------|-----|
| case001 | 1250.5  | 3420.8  | 890.2   | ... |
| case002 | 1180.3  | 3350.1  | 910.5   | ... |

### 5. Volume_Pred

Volume in cubic millimeters for each class in each sample (from prediction).

| sample  | class_0 | class_1 | class_2 | ... |
|---------|---------|---------|---------|-----|
| case001 | 1245.2  | 3380.5  | 885.1   | ... |
| case002 | 1175.8  | 3320.3  | 915.2   | ... |
