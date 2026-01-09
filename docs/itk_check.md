# itk_check

Check ITK image-label sample pairs to verify they meet size and spacing requirements.

## Usage

```bash
itk_check <mode> <sample_folder> [options]
```

## Modes

- **check**: Validate image/label pairs against size/spacing rules and report non-conforming samples (no file changes)
- **delete**: Remove image and label files for samples that fail validation
- **copy**: Copy valid image/label pairs to the specified output directory
- **symlink**: Create symbolic links for valid image/label pairs in the output directory

## Parameters

- `sample_folder`: Root folder containing `image/` and `label/` subfolders
- `-o, --output OUT`: Output directory (required for `copy` and `symlink` modes)
- `--min-size Z Y X`: Minimum size per dimension (three integers; -1 = ignore)
- `--max-size Z Y X`: Maximum size per dimension (three integers; -1 = ignore)
- `--min-spacing Z Y X`: Minimum spacing per dimension (three floats; -1 = ignore)
- `--max-spacing Z Y X`: Maximum spacing per dimension (three floats; -1 = ignore)
- `--same-spacing A B`: Two dimensions (X|Y|Z) that must have equal spacing
- `--same-size A B`: Two dimensions (X|Y|Z) that must have equal size
- `--mp`: Enable multiprocessing

## Examples

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
