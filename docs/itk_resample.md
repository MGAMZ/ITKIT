# itk_resample

Resample ITK image-label sample pairs to a target spacing or size.

## Usage

```bash
itk_resample <field> <source_folder> <dest_folder> [options]
```

## Field Types

- **image**: For image data (uses linear interpolation, preserves data type)
- **label**: For label/segmentation data (uses nearest neighbor interpolation)
- **dataset**: Processes both `image/` and `label/` subfolders with appropriate settings

## Parameters

- `source_folder`: Folder containing source image files
- `dest_folder`: Destination folder for resampled files (created if missing)
- `--spacing Z Y X`: Target spacing per dimension (ZYX order). Use -1 to ignore a dimension
- `--size Z Y X`: Target size per dimension (ZYX order). Use -1 to ignore a dimension
- `--target-folder PATH`: Folder of reference images (mutually exclusive with `--spacing/--size`)
- `-r, --recursive`: Recursively process subdirectories, preserving layout
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes for multiprocessing

## Output

- Resampled files in `dest_folder`
- `resample_configs.json`: Configuration used for resampling
- `meta.json`: Metadata for the resampled dataset

## Examples

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
