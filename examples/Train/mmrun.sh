set -e

# mmrun 1.0 \
#     --models SegFormer3D \
#     --work-dir-root tmp/workdir \
#     --test-work-dir-root tmp/testdir \
#     --config-root examples/Train

mmrun 1.1 \
    --models SegFormer3D \
    --work-dir-root tmp/workdir \
    --test-work-dir-root tmp/testdir \
    --config-root examples/Train
