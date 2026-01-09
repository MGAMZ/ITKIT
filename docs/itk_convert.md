# itk_convert

Convert ITKIT datasets between different formats and frameworks.

## Subcommands

- **format**: Convert medical image file formats
- **monai**: Convert to MONAI Decathlon format
- **torchio**: Convert to TorchIO format

## Format Conversion

Convert medical image files between different formats while preserving metadata.

### Usage

```bash
itk_convert format <target_format> <source_folder> <dest_folder> [options]
```

### Supported Formats

- `mha`: MetaImage (single file)
- `mhd`: MetaImage Header (with separate .raw file)
- `nii.gz`: Compressed NIfTI
- `nii`: NIfTI (uncompressed)
- `nrrd`: Nearly Raw Raster Data

### Parameters

- `target_format`: Target file format
- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

### Examples

```bash
# Convert MHA to compressed NIfTI
itk_convert format nii.gz /data/mha_dataset /data/nifti_dataset

# Convert to NRRD with multiprocessing
itk_convert format nrrd /data/input /data/output --mp --workers 8

# Convert MHD to MHA
itk_convert format mha /data/mhd_dataset /data/mha_dataset
```

## MONAI Conversion

Convert ITKIT dataset to MONAI Decathlon format.

### Usage

```bash
itk_convert monai <source_folder> <dest_folder> [options]
```

### Parameters

- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset in MONAI format
- `--name`: Dataset name for the manifest file (default: `ITKITDataset`)
- `--description`: Dataset description for the manifest file
- `--modality`: Primary imaging modality (default: `CT`)
- `--split`: Which split to treat the data as: `train` | `val` | `test` | `all` (default: `train`)
- `--labels`: Label names in order (e.g., `background liver tumor`). Index 0 is background
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

### Output

- Converted files in `.nii.gz` format
- `dataset.json` manifest file
- `meta.json` ITKIT-style metadata

### Examples

```bash
itk_convert monai /data/itkit_dataset /data/monai_dataset \
    --name MyDataset \
    --modality CT \
    --labels background liver tumor \
    --mp
```

## TorchIO Conversion

Convert ITKIT dataset to TorchIO format.

### Usage

```bash
itk_convert torchio <source_folder> <dest_folder> [options]
```

### Parameters

- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset in TorchIO format
- `--no-csv`: Skip creating `subjects.csv` manifest file
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

### Output

- Converted files in `.nii.gz` format
- `subjects.csv` manifest file (unless `--no-csv` is specified)
- `meta.json` ITKIT-style metadata

### Examples

```bash
itk_convert torchio /data/itkit_dataset /data/torchio_dataset --mp
```
