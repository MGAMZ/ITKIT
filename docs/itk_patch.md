# itk_patch

Extract patches from ITK image-label sample pairs for training.

## Usage

```bash
itk_patch <src_folder> <dst_folder> --patch-size SIZE --patch-stride STRIDE [options]
```

## Parameters

- `src_folder`: Source root containing `image/` and `label/` subfolders
- `dst_folder`: Destination root to save patches
- `--patch-size`: Patch size as single int or three ints (Z Y X)
- `--patch-stride`: Patch stride as single int or three ints (Z Y X)
- `--minimum-foreground-ratio`: Minimum label foreground ratio to keep a patch (float, default 0.0)
- `--keep-empty-label-prob`: Probability to keep patches with only background (0.0-1.0)
- `--still-save-when-no-label`: If set and label missing, save patches regardless
- `--mp`: Use multiprocessing to process cases in parallel

## Output

- Patches saved under `dst_folder/<case_name>/` with image and label patch files
- `crop_meta.json`: Summary of extraction and available annotations

## Examples

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
