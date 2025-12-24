set -e

for example_id in "0.0" "0.1" "0.2"
do
mmrun ${example_id} \
    --models SegFormer3D \
    --work-dir-root examples/workdir \
    --test-work-dir-root examples/testdir \
    --config-root examples/configs
done
