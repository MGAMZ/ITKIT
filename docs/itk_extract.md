# itk_extract

Extract specified classes from ITK semantic segmentation maps.

## Usage

```bash
itk_extract <source_folder> <dest_folder> <mappings...> [options]
```

## Parameters

- `source_folder`: Folder containing source images
- `dest_folder`: Destination folder to save extracted label files (created if missing)
- `mappings`: One or more label mappings in `"source:target"` format (e.g., `"1:0"` `"5:1"`)
- `-r, --recursive`: Recursively process subdirectories and preserve relative paths
- `--mp`: Enable multiprocessing
- `--workers N`: Number of worker processes for multiprocessing

## Output

- Remapped label files written to `dest_folder` (extensions normalized to `.mha`)
- `extract_meta.json`: Per-sample metadata
- `extract_configs.json`: Configuration used

## Examples

```bash
# Extract liver (label 1) and tumor (label 5), renumber to 0 and 1
itk_extract /data/labels /data/extracted "1:0" "5:1" --mp

# Extract specific organs from multi-organ segmentation
itk_extract /data/multi_organ /data/liver_kidney \
    "1:1" "2:2" \
    --recursive --mp
```
