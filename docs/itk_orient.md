# itk_orient

Orient ITK image-label sample pairs to a specified orientation.

## Usage

```bash
itk_orient <src_dir> <dst_dir> <orient> [options]
```

## Parameters

- `src_dir`: Source directory containing `.mha` files (recursive scan)
- `dst_dir`: Destination directory (preserves relative directory structure; must differ from `src_dir`)
- `orient`: Target orientation string for SimpleITK.DICOMOrient (e.g., `LPI`, `RAS`)
- `--mp`: Use multiprocessing to convert files in parallel

## Common Orientations

- **LPI**: Left-Posterior-Inferior (common in medical imaging)
- **RAS**: Right-Anterior-Superior (neuroimaging standard)
- **LPS**: Left-Posterior-Superior

## Notes

- Skips files already present in `dst_dir`
- Preserves folder layout
- Writes converted `.mha` files to `dst_dir`

## Examples

```bash
# Orient to LPI
itk_orient /data/source /data/oriented LPI --mp

# Orient to RAS (neuroimaging standard)
itk_orient /data/source /data/ras_oriented RAS
```
