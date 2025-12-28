# ITK Preprocessing Tools

ITKIT provides comprehensive command-line tools for medical image preprocessing. Each tool is designed for a specific operation and follows a consistent interface pattern.

## General Notes

- **Coordinate Order**: All dimension arguments use **Z, Y, X** order (Z→0, Y→1, X→2)
- **Help**: Use `--help` with any command to see detailed usage information
- **Multiprocessing**: Most commands support `--mp` flag for parallel processing
- **Progress**: Commands display progress bars using tqdm

## itk_check

Check ITK image-label sample pairs to verify they meet size and spacing requirements.

### Usage

```bash
itk_check <mode> <sample_folder> [options]
```

### Modes

- **check**: Validate image/label pairs against size/spacing rules and report non-conforming samples (no file changes)
- **delete**: Remove image and label files for samples that fail validation
- **copy**: Copy valid image/label pairs to the specified output directory
- **symlink**: Create symbolic links for valid image/label pairs in the output directory

### Parameters

- `sample_folder`: Root folder containing `image/` and `label/` subfolders
- `-o, --output OUT`: Output directory (required for `copy` and `symlink` modes)
- `--min-size Z Y X`: Minimum size per dimension (three integers; -1 = ignore)
- `--max-size Z Y X`: Maximum size per dimension (three integers; -1 = ignore)
- `--min-spacing Z Y X`: Minimum spacing per dimension (three floats; -1 = ignore)
- `--max-spacing Z Y X`: Maximum spacing per dimension (three floats; -1 = ignore)
- `--same-spacing A B`: Two dimensions (X|Y|Z) that must have equal spacing
- `--same-size A B`: Two dimensions (X|Y|Z) that must have equal size
- `--mp`: Enable multiprocessing

### Examples

```bash
# Check dataset without modifications
itk_check check /data/dataset --min-size 32 32 32

# Copy valid samples to new location
itk_check copy /data/dataset --output /data/valid_dataset \
    --min-spacing 0.5 0.5 0.5 \
    --max-spacing 2.0 2.0 2.0

# Check that X and Y spacing are equal
itk_check check /data/dataset --same-spacing X Y
```

---

## itk_resample

Resample ITK image-label sample pairs to a target spacing or size.

### Usage

```bash
itk_resample <field> <source_folder> <dest_folder> [options]
```

### Field Types

- **image**: For image data (uses linear interpolation, preserves data type)
- **label**: For label/segmentation data (uses nearest neighbor interpolation)
- **dataset**: Processes both `image/` and `label/` subfolders with appropriate settings

### Parameters

- `source_folder`: Folder containing source image files
- `dest_folder`: Destination folder for resampled files (created if missing)
- `--spacing Z Y X`: Target spacing per dimension (ZYX order). Use -1 to ignore a dimension
- `--size Z Y X`: Target size per dimension (ZYX order). Use -1 to ignore a dimension
- `--target-folder PATH`: Folder of reference images (mutually exclusive with `--spacing/--size`)
- `-r, --recursive`: Recursively process subdirectories, preserving layout
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes for multiprocessing

### Output

- Resampled files in `dest_folder`
- `resample_configs.json`: Configuration used for resampling
- `meta.json`: Metadata for the resampled dataset

### Examples

```bash
# Resample entire dataset to 1.0mm isotropic spacing
itk_resample dataset /data/source /data/resampled \
    --spacing 1.0 1.0 1.0 --mp

# Resample only in-plane, keep Z spacing
itk_resample image /data/source /data/resampled \
    --spacing -1 0.5 0.5

# Resample to match reference dataset
itk_resample dataset /data/source /data/resampled \
    --target-folder /data/reference --mp
```

---

## itk_orient

Orient ITK image-label sample pairs to a specified orientation.

### Usage

```bash
itk_orient <src_dir> <dst_dir> <orient> [options]
```

### Parameters

- `src_dir`: Source directory containing `.mha` files (recursive scan)
- `dst_dir`: Destination directory (preserves relative directory structure; must differ from `src_dir`)
- `orient`: Target orientation string for SimpleITK.DICOMOrient (e.g., `LPI`, `RAS`)
- `--mp`: Use multiprocessing to convert files in parallel

### Common Orientations

- **LPI**: Left-Posterior-Inferior (common in medical imaging)
- **RAS**: Right-Anterior-Superior (neuroimaging standard)
- **LPS**: Left-Posterior-Superior

### Notes

- Skips files already present in `dst_dir`
- Preserves folder layout
- Writes converted `.mha` files to `dst_dir`

### Examples

```bash
# Orient to LPI
itk_orient /data/source /data/oriented LPI --mp

# Orient to RAS (neuroimaging standard)
itk_orient /data/source /data/ras_oriented RAS
```

---

## itk_patch

Extract patches from ITK image-label sample pairs for training.

### Usage

```bash
itk_patch <src_folder> <dst_folder> --patch-size SIZE --patch-stride STRIDE [options]
```

### Parameters

- `src_folder`: Source root containing `image/` and `label/` subfolders
- `dst_folder`: Destination root to save patches
- `--patch-size`: Patch size as single int or three ints (Z Y X)
- `--patch-stride`: Patch stride as single int or three ints (Z Y X)
- `--minimum-foreground-ratio`: Minimum label foreground ratio to keep a patch (float, default 0.0)
- `--keep-empty-label-prob`: Probability to keep patches with only background (0.0-1.0)
- `--still-save-when-no-label`: If set and label missing, save patches regardless
- `--mp`: Use multiprocessing to process cases in parallel

### Output

- Patches saved under `dst_folder/<case_name>/` with image and label patch files
- `crop_meta.json`: Summary of extraction and available annotations

### Examples

```bash
# Extract 96x96x96 patches with 48-voxel stride
itk_patch /data/dataset /data/patches \
    --patch-size 96 96 96 \
    --patch-stride 48 48 48 \
    --mp

# Extract patches with foreground filtering
itk_patch /data/dataset /data/patches \
    --patch-size 128 128 128 \
    --patch-stride 64 64 64 \
    --minimum-foreground-ratio 0.1 \
    --keep-empty-label-prob 0.2 \
    --mp
```

---

## itk_aug

Perform data augmentation on ITK image files.

### Usage

```bash
itk_aug <img_folder> <lbl_folder> [options]
```

### Parameters

- `img_folder`: Folder with source image `.mha` files
- `lbl_folder`: Folder with source label `.mha` files
- `-oimg, --out-img-folder OUT_IMG`: Optional folder to save augmented images
- `-olbl, --out-lbl-folder OUT_LBL`: Optional folder to save augmented labels
- `-n, --num N`: Number of augmented samples to generate per source sample
- `--mp`: Enable multiprocessing
- `--random-rot Z Y X`: Max random rotation degrees for Z Y X axes (three ints, order Z, Y, X)

### Notes

- Only files present in both `img_folder` and `lbl_folder` are processed
- Augmented files are written only if corresponding output folders are provided
- Currently supports: **RandomRotate3D**

### Examples

```bash
# Generate 5 augmented samples per input with random rotation
itk_aug /data/images /data/labels \
    -oimg /data/aug_images \
    -olbl /data/aug_labels \
    -n 5 \
    --random-rot 15 15 15 \
    --mp
```

---

## itk_extract

Extract specified classes from ITK semantic segmentation maps.

### Usage

```bash
itk_extract <source_folder> <dest_folder> <mappings...> [options]
```

### Parameters

- `source_folder`: Folder containing source images
- `dest_folder`: Destination folder to save extracted label files (created if missing)
- `mappings`: One or more label mappings in `"source:target"` format (e.g., `"1:0"` `"5:1"`)
- `-r, --recursive`: Recursively process subdirectories and preserve relative paths
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes for multiprocessing

### Output

- Remapped label files written to `dest_folder` (extensions normalized to `.mha`)
- `extract_meta.json`: Per-sample metadata
- `extract_configs.json`: Configuration used

### Examples

```bash
# Extract liver (label 1) and tumor (label 5), renumber to 0 and 1
itk_extract /data/labels /data/extracted "1:0" "5:1" --mp

# Extract specific organs from multi-organ segmentation
itk_extract /data/multi_organ /data/liver_kidney \
    "1:1" "2:2" \
    --recursive --mp
```

---

## itk_convert

Convert ITKIT datasets between different formats and frameworks.

### Subcommands

- **format**: Convert medical image file formats
- **monai**: Convert to MONAI Decathlon format
- **torchio**: Convert to TorchIO format

### Format Conversion

Convert medical image files between different formats while preserving metadata.

#### Usage

```bash
itk_convert format <target_format> <source_folder> <dest_folder> [options]
```

#### Supported Formats

- `mha`: MetaImage (single file)
- `mhd`: MetaImage Header (with separate .raw file)
- `nii.gz`: Compressed NIfTI
- `nii`: NIfTI (uncompressed)
- `nrrd`: Nearly Raw Raster Data

#### Parameters

- `target_format`: Target file format
- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

#### Examples

```bash
# Convert MHA to compressed NIfTI
itk_convert format nii.gz /data/mha_dataset /data/nifti_dataset

# Convert to NRRD with multiprocessing
itk_convert format nrrd /data/input /data/output --mp --workers 8

# Convert MHD to MHA
itk_convert format mha /data/mhd_dataset /data/mha_dataset
```

### MONAI Conversion

Convert ITKIT dataset to MONAI Decathlon format.

#### Usage

```bash
itk_convert monai <source_folder> <dest_folder> [options]
```

#### Parameters

- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset in MONAI format
- `--name`: Dataset name for the manifest file (default: `ITKITDataset`)
- `--description`: Dataset description for the manifest file
- `--modality`: Primary imaging modality (default: `CT`)
- `--split`: Which split to treat the data as: `train` | `val` | `test` | `all` (default: `train`)
- `--labels`: Label names in order (e.g., `background liver tumor`). Index 0 is background
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

#### Output

- Converted files in `.nii.gz` format
- `dataset.json` manifest file
- `meta.json` ITKIT-style metadata

#### Examples

```bash
itk_convert monai /data/itkit_dataset /data/monai_dataset \
    --name MyDataset \
    --modality CT \
    --labels background liver tumor \
    --mp
```

### TorchIO Conversion

Convert ITKIT dataset to TorchIO format.

#### Usage

```bash
itk_convert torchio <source_folder> <dest_folder> [options]
```

#### Parameters

- `source_folder`: Path to ITKIT dataset
- `dest_folder`: Path to output dataset in TorchIO format
- `--no-csv`: Skip creating `subjects.csv` manifest file
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes

#### Output

- Converted files in `.nii.gz` format
- `subjects.csv` manifest file (unless `--no-csv` is specified)
- `meta.json` ITKIT-style metadata

#### Examples

```bash
itk_convert torchio /data/itkit_dataset /data/torchio_dataset --mp
```

---

## Best Practices

1. **Use multiprocessing**: Add `--mp` flag for large datasets to speed up processing
2. **Check before processing**: Always run `itk_check` before other operations
3. **Preserve originals**: Work on copies of your data, never modify originals
4. **Pipeline operations**: Chain commands to create preprocessing pipelines
5. **Validate outputs**: Check a few samples manually after each processing step
6. **Use consistent spacing**: Resample all data to the same spacing for training

## Next Steps

- [Dataset Structure](dataset_structure.md) - Understand dataset requirements
- [Framework Integration](framework_integration.md) - Learn about deep learning integration
