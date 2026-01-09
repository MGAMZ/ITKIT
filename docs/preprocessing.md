# ITK Preprocessing Tools

ITKIT provides comprehensive command-line tools for medical image preprocessing. Each tool is designed for a specific operation and follows a consistent interface pattern.

## General Notes

- **Coordinate Order**: All dimension arguments use **Z, Y, X** order (Z→0, Y→1, X→2)
- **Help**: Use '--help' with any command to see detailed usage information
- **Multiprocessing**: Most commands support '--mp' flag for parallel processing
- **Progress**: Commands display progress bars using tqdm

## Best Practices

1. **Use multiprocessing**: Add '--mp' flag for large datasets to speed up processing
2. **Check before processing**: Always run 'itk_check' before other operations
3. **Preserve originals**: Work on copies of your data, never modify originals
4. **Pipeline operations**: Chain commands to create preprocessing pipelines
5. **Validate outputs**: Check a few samples manually after each processing step
6. **Use consistent spacing**: Resample all data to the same spacing for training
