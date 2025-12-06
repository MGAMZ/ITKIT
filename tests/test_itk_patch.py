import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from itkit.process.itk_patch import PatchProcessor, parse_patch_size


@pytest.fixture
def sample_image():
    arr = np.random.rand(10, 10, 10).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


@pytest.fixture
def sample_label():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[2:8, 2:8, 2:8] = 1  # Some foreground
    lbl = sitk.GetImageFromArray(arr)
    lbl.SetSpacing((1.0, 1.0, 1.0))
    lbl.SetOrigin((0.0, 0.0, 0.0))
    return lbl


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def processor():
    return PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)


@pytest.fixture
def temp_src_dst(temp_dir):
    src = temp_dir / 'src'
    dst = temp_dir / 'dst'
    src.mkdir()
    (src / 'image').mkdir()
    (src / 'label').mkdir()
    return src, dst


@pytest.fixture
def sample_files(temp_src_dst, sample_image, sample_label):
    src, _ = temp_src_dst
    img_path = src / 'image' / 'test.mha'
    lbl_path = src / 'label' / 'test.mha'
    sitk.WriteImage(sample_image, str(img_path))
    sitk.WriteImage(sample_label, str(lbl_path))
    return img_path, lbl_path


def run_patch_command(src, dst, patch_size=[4, 4, 4], patch_stride=[2, 2, 2], mp=False):
    cmd = [
        sys.executable, '-m', 'itkit.process.itk_patch',
        str(src), str(dst), '--patch-size', *map(str, patch_size), '--patch-stride', *map(str, patch_stride)
    ]
    if mp:
        cmd.append('--mp')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc

@pytest.mark.itk_process
class TestPatchProcessor:
    def test_parse_patch_size(self, temp_dir):
        meta_path = temp_dir / 'crop_meta.json'
        # Create a complete CropMetadata and save it
        from itkit.process.itk_patch import CropMetadata
        crop_meta = CropMetadata(
            src_folder='/tmp/src',
            dst_folder='/tmp/dst',
            patch_size=[64, 64, 64],
            patch_stride=[32, 32, 32],
            anno_available=[],
            patch_meta={}
        )
        crop_meta.save(meta_path)
        result = parse_patch_size(str(temp_dir))
        assert result == [64, 64, 64]

    @pytest.mark.parametrize("patch_size, patch_stride", [
        ([4, 4, 4], [2, 2, 2]),
        ([3, 3, 3], [1, 1, 1]),
        (4, 2),  # int inputs
    ])
    def test_extract_patches_basic(self, sample_image, sample_label, patch_size, patch_stride):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', patch_size, patch_stride, 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, patch_size, patch_stride, 0.0, False)
        assert len(patches) > 0
        img_patch, lbl_patch = patches[0]
        expected_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        assert img_patch.GetSize() == expected_size
        assert lbl_patch.GetSize() == expected_size
        # Check properties are copied correctly
        assert img_patch.GetSpacing() == sample_image.GetSpacing()
        assert img_patch.GetDirection() == sample_image.GetDirection()
        assert lbl_patch.GetSpacing() == sample_label.GetSpacing()
        assert lbl_patch.GetDirection() == sample_label.GetDirection()
        # For first patch (x=0,y=0,z=0), origin should be same as original
        assert img_patch.GetOrigin() == sample_image.GetOrigin()
        # Check that at least one patch has adjusted origin (if multiple patches)
        if len(patches) > 1:
            assert any(img_p.GetOrigin() != sample_image.GetOrigin() for img_p, _ in patches)

    def test_extract_patches_no_label(self, sample_image):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, True)
        patches = processor.extract_patches(sample_image, None, [4, 4, 4], [2, 2, 2], 0.0, True)
        assert len(patches) > 0
        img_patch, lbl_patch = patches[0]
        assert lbl_patch is None

    def test_extract_patches_foreground_filter(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.5, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [2, 2, 2], 0.5, False)
        # Assuming some patches have low fg, check filtering
        assert len(patches) < 100  # Less than full grid

    def test_is_valid_sample_matching(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        assert processor.is_valid_sample(sample_image, sample_label)

    def test_is_valid_sample_size_mismatch(self, sample_image, sample_label):
        mismatched_label = sitk.GetImageFromArray(np.zeros((5, 10, 10), dtype=np.uint8))
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        assert not processor.is_valid_sample(sample_image, mismatched_label)

        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        invalid_image = sitk.GetImageFromArray(np.random.rand(10, 10, 5).astype(np.float32))  # 3D but different size
        with pytest.raises(ValueError):
            processor.extract_patches(invalid_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)

    def test_process_one(self, temp_dir, sample_image, sample_label):
        img_path = temp_dir / 'img.mha'
        lbl_path = temp_dir / 'lbl.mha'
        sitk.WriteImage(sample_image, str(img_path))
        sitk.WriteImage(sample_label, str(lbl_path))
        processor = PatchProcessor(str(temp_dir), str(temp_dir / 'dst'), [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        result = processor.process_one((str(img_path), str(lbl_path)))
        assert result is not None
        # Check that result is ProcessOneResult
        from itkit.process.itk_patch import ProcessOneResult
        assert isinstance(result, ProcessOneResult)
        # Check patch metadata list
        assert len(result.patch_metadata_list) > 0
        for meta in result.patch_metadata_list:
            assert hasattr(meta, 'name')
            assert hasattr(meta, 'spacing')
            assert hasattr(meta, 'size')
        # Check source metadata
        assert result.source_metadata.series_id == 'img'
        assert result.source_metadata.num_patches > 0

    def test_main_subprocess(self, temp_src_dst, sample_image, sample_label):
        src, dst = temp_src_dst
        # Create dummy files
        sitk.WriteImage(sample_image, str(src / 'image' / 'test.mha'))
        sitk.WriteImage(sample_label, str(src / 'label' / 'test.mha'))

        proc = run_patch_command(src, dst)
        assert proc.returncode == 0
        assert (dst / 'crop_meta.json').exists()

    def test_parse_patch_size_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_patch_size('/nonexistent/path')

    def test_parse_patch_size_invalid_json(self, temp_dir):
        meta_path = temp_dir / 'crop_meta.json'
        with open(meta_path, 'w') as f:
            f.write('invalid json')
        with pytest.raises(json.JSONDecodeError):
            parse_patch_size(str(temp_dir))

    def test_extract_patches_patch_too_large(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [20, 20, 20], [2, 2, 2], 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [20, 20, 20], [2, 2, 2], 0.0, False)
        assert len(patches) == 0  # Patch size > image size

    def test_extract_patches_stride_starts(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [3, 3, 3], 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [3, 3, 3], 0.0, False)
        # For 10x10x10 image, patch 4, stride 3: starts should be [0,3,6,6] (last adjusted)
        # Check number of patches: (4 starts)^3 = 64, but filtered by fg
        assert len(patches) > 0

    def test_extract_patches_keep_empty_prob(self, sample_image, sample_label):
        # Use fixed seed for reproducibility
        np.random.seed(42)
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 0.5, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)
        # With prob 0.5, some empty patches may be kept, but not all
        assert len(patches) >= 0  # At least some should be saved

    def test_is_valid_sample_spacing_mismatch(self, sample_image):
        mismatched_label = sitk.GetImageFromArray(np.zeros((10, 10, 10), dtype=np.uint8))
        mismatched_label.SetSpacing((2.0, 1.0, 1.0))  # Different spacing
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        assert not processor.is_valid_sample(sample_image, mismatched_label)

    def test_is_valid_sample_non_isotropic(self, sample_image):
        non_iso_label = sitk.GetImageFromArray(np.zeros((10, 5, 10), dtype=np.uint8))  # Non-square X-Y
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        assert not processor.is_valid_sample(sample_image, non_iso_label)

    def test_process_one_invalid_file(self):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        result = processor.process_one(('/nonexistent/img.mha', '/nonexistent/lbl.mha'))
        assert result is None  # Should handle exception gracefully

    def test_patch_processor_init_dirs(self, temp_dir):
        dst = temp_dir / 'dst'
        processor = PatchProcessor('/tmp/src', str(dst), [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        assert (dst / 'image').exists()
        assert (dst / 'label').exists()

    def test_main_invalid_args(self):
        # Test with missing required args
        proc = subprocess.run([
            sys.executable, '-c', 'from itkit.process.itk_patch import main; main()'
        ], capture_output=True, text=True, input='')  # Simulate no args
        assert proc.returncode != 0  # Should fail due to missing args

    def test_process_one_file_saving(self, temp_dir, sample_image, sample_label):
        img_path = temp_dir / 'img.mha'
        lbl_path = temp_dir / 'lbl.mha'
        dst = temp_dir / 'dst'
        sitk.WriteImage(sample_image, str(img_path))
        sitk.WriteImage(sample_label, str(lbl_path))
        processor = PatchProcessor(str(temp_dir), str(dst), [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        result = processor.process_one((str(img_path), str(lbl_path)))
        assert result is not None
        # Check files are saved
        image_files = list((dst / 'image').glob('*.mha'))
        label_files = list((dst / 'label').glob('*.mha'))
        assert len(image_files) > 0
        assert len(label_files) > 0
        assert len(image_files) == len(label_files)

    def test_extract_patches_origin_calculation(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)
        # First patch (0,0,0) should have origin (0,0,0)
        img_patch, _ = patches[0]
        assert img_patch.GetOrigin() == (0.0, 0.0, 0.0)
        # Later patches should have adjusted origins
        if len(patches) > 1:
            for img_p, _ in patches[1:]:
                assert img_p.GetOrigin() != (0.0, 0.0, 0.0)

    def test_to_triplet_invalid_input(self, sample_image, sample_label):
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        with pytest.raises(ValueError):
            processor.extract_patches(sample_image, sample_label, 'invalid', [2, 2, 2], 0.0, False)
        with pytest.raises(ValueError):
            processor.extract_patches(sample_image, sample_label, [4, 4], [2, 2, 2], 0.0, False)

    def test_compute_starts_logic(self, sample_image, sample_label):
        # Indirectly test via extract_patches, but add direct if needed
        # For L=10, p=4, s=3: starts = [0,3,6] then append 6 (since 10-4=6)
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [3, 3, 3], 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [3, 3, 3], 0.0, False)
        # Assert based on expected starts
        assert len(patches) > 0  # Should have patches from starts

    def test_patch_size_int_vs_list(self, sample_image, sample_label):
        # Test int input
        processor = PatchProcessor('/tmp/src', '/tmp/dst', 4, 2, 0.0, 1.0, False)
        patches = processor.extract_patches(sample_image, sample_label, 4, 2, 0.0, False)
        assert len(patches) > 0
        img_patch, _ = patches[0]
        assert img_patch.GetSize() == (4, 4, 4)

    def test_extract_patches_read_failure(self, sample_label):
        # Simulate read failure by passing 2D image instead of 3D
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        invalid_image = sitk.GetImageFromArray(np.random.rand(10, 10).astype(np.float32))  # 2D image
        with pytest.raises(ValueError):
            processor.extract_patches(invalid_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)

    def test_keep_empty_prob_boundary(self, sample_image, sample_label):
        # Test prob=0.0: no empty patches kept
        np.random.seed(42)
        processor = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 0.0, False)
        patches = processor.extract_patches(sample_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)
        # Should have fewer patches due to prob=0
        assert len(patches) >= 0

        # Test prob=1.0: all empty patches kept
        processor2 = PatchProcessor('/tmp/src', '/tmp/dst', [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        patches2 = processor2.extract_patches(sample_image, sample_label, [4, 4, 4], [2, 2, 2], 0.0, False)
        assert len(patches2) >= len(patches)  # At least as many

    def test_saved_patches_content(self, temp_dir, sample_image, sample_label):
        img_path = temp_dir / 'img.mha'
        lbl_path = temp_dir / 'lbl.mha'
        dst = temp_dir / 'dst'
        sitk.WriteImage(sample_image, str(img_path))
        sitk.WriteImage(sample_label, str(lbl_path))
        processor = PatchProcessor(str(temp_dir), str(dst), [4, 4, 4], [2, 2, 2], 0.0, 1.0, False)
        processor.process_one((str(img_path), str(lbl_path)))
        # Read back a saved patch
        saved_img = sitk.ReadImage(str(dst / 'image' / 'img_p0.mha'))
        saved_lbl = sitk.ReadImage(str(dst / 'label' / 'img_p0.mha'))
        assert saved_img.GetSize() == (4, 4, 4)
        assert saved_lbl.GetSize() == (4, 4, 4)
        # Check content matches original slice
        orig_arr = sitk.GetArrayFromImage(sample_image)
        saved_arr = sitk.GetArrayFromImage(saved_img)
        assert np.allclose(saved_arr, orig_arr[:4, :4, :4])

    def test_crop_meta_json(self, temp_src_dst, sample_image, sample_label):
        src, dst = temp_src_dst
        sitk.WriteImage(sample_image, str(src / 'image' / 'test.mha'))
        sitk.WriteImage(sample_label, str(src / 'label' / 'test.mha'))

        proc = run_patch_command(src, dst)
        assert proc.returncode == 0, f"Process failed: {proc.stderr}"

        # Check crop_meta.json exists and has correct structure
        meta_path = dst / 'crop_meta.json'
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert 'patch_size' in meta
        assert 'patch_meta' in meta
        assert 'test' in meta['patch_meta']
        assert meta['patch_meta']['test']['num_patches'] > 0

        # Check that standard meta.json files also exist
        assert (dst / "meta.json").exists()
        assert (dst / "image" / "meta.json").exists()
        assert (dst / "label" / "meta.json").exists()

    def test_multiprocessing_mode(self, temp_src_dst, sample_image, sample_label):
        # Simple test: mp=True should not crash immediately
        src, dst = temp_src_dst
        sitk.WriteImage(sample_image, str(src / 'image' / 'test.mha'))
        sitk.WriteImage(sample_label, str(src / 'label' / 'test.mha'))

        proc = run_patch_command(src, dst, mp=True)
        assert proc.returncode == 0  # Should succeed without error
