# itk_aug

Perform data augmentation on ITK image files.

## Usage

```bash
itk_aug <img_folder> <lbl_folder> [options]
```

## Parameters

- `img_folder`: Folder with source image `.mha` files
- `lbl_folder`: Folder with source label `.mha` files
- `-oimg, --out-img-folder OUT_IMG`: Optional folder to save augmented images
- `-olbl, --out-lbl-folder OUT_LBL`: Optional folder to save augmented labels
- `-n, --num N`: Number of augmented samples to generate per source sample
- `--mp`: Enable multiprocessing
- `--random-rot Z Y X`: Max random rotation degrees for Z Y X axes (three ints, order Z, Y, X)

## Notes

- Only files present in both `img_folder` and `lbl_folder` are processed
- Augmented files are written only if corresponding output folders are provided
- Currently supports: **RandomRotate3D**

## Examples

```bash
# Generate 5 augmented samples per input with random rotation
itk_aug /data/images /data/labels \
    -oimg /data/aug_images \
    -olbl /data/aug_labels \
    -n 5 \
    --random-rot 15 15 15 \
    --mp
```
