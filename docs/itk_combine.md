# itk_combine

Combine multiple label folders by intersecting filenames and merging labels according to ordered mapping rules. This tool is useful when you have multiple specialized segmentations for the same cases and want to create a unified label map.

## Usage

```bash
itk_combine --source <name>=<folder> --map <mapping_rule> <dest_folder> [options]
```

## Parameters

- `--source`: Specify a label source in the format `name=/path/to/folder`. Can be specified multiple times for different sources.
- `--map`: Specify a mapping rule in the format `<source_name>:<source_labels>-><target_label>`.
  - `<source_name>` must match one of the names defined in `--source`.
  - `<source_labels>` can be a single integer or a comma-separated list of integers.
  - Multiple `--map` rules are allowed. **Priority is determined by order**: the first rule that matches a voxel determines its value in the output.
- `dest_folder`: Destination folder for the combined label files.
- `--mp`: Enable multiprocessing.
- `--workers`: Number of worker processes (defaults to half of CPU cores).

## Mapping Priority and Logic

1. **Intersection**: Only files that exist in **all** specified source folders (with the same base name) will be processed.

2. **Validation**: For each file, the tool ensures that the image size and spacing are identical across all sources. If a mismatch is found, the process will fail.

3. **Merging**:

   - The output label map is initialized to 0 (Background).
   - Rules are applied sequentially in the order they appear in the command line.
   - Once a voxel is assigned a non-zero value, it will not be overwritten by subsequent rules. This allows for clear priority management between overlapping sources.

## Example

Suppose you have:

- `Source A`: Organ segmentations (1: Liver, 2: Spleen)
- `Source B`: Tumor segmentations (1: Liver Tumor)

To combine them into a single map where Background=0, Liver=1, Spleen=2, and Liver Tumor=3 (with tumor taking priority over the organ label):

```bash
itk_combine \
    --source organs=/path/to/organs \
    --source tumors=/path/to/tumors \
    --map tumors:1->3 \
    --map organs:1->1 \
    --map organs:2->2 \
    /path/to/combined_output \
    --mp
```

## Output

- Combined label maps (normalized to `.mha` format and `uint8` data type).
- `meta.json`: Standard ITKIT metadata file containing size, spacing, origin, and unique classes for each combined file.
