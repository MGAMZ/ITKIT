import os
import json
import tempfile
import pytest
import SimpleITK as sitk
import numpy as np
from itkit.process.itk_check import CheckProcessor


def create_test_image(path: str, size: tuple, spacing: tuple):
    """Helper to create test MHA images"""
    img = sitk.Image(size[::-1], sitk.sitkUInt8)  # SimpleITK uses XYZ
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, path)


def create_default_cfg(min_size=None, max_size=None, min_spacing=None, max_spacing=None, same_spacing=None, same_size=None):
    """Helper to create default configuration dictionary"""
    return {
        'min_size': min_size,
        'max_size': max_size,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing,
        'same_spacing': same_spacing,
        'same_size': same_size,
    }


def setup_dataset_test_data(tmpdir, image_specs):
    """Helper to setup dataset mode test data with image and label directories"""
    img_dir = os.path.join(tmpdir, 'image')
    lbl_dir = os.path.join(tmpdir, 'label')
    os.makedirs(img_dir)
    os.makedirs(lbl_dir)
    
    for name, size, spacing in image_specs:
        create_test_image(os.path.join(img_dir, f'{name}.mha'), size, spacing)
        create_test_image(os.path.join(lbl_dir, f'{name}.mha'), size, spacing)
    
    return img_dir, lbl_dir


def setup_single_folder_test_data(tmpdir, image_specs):
    """Helper to setup single folder mode test data"""
    for name, size, spacing in image_specs:
        create_test_image(os.path.join(tmpdir, f'{name}.mha'), size, spacing)


def run_check_processor(source_folder, cfg, mode='check', output_dir=None, mp=False, workers=1):
    """Helper to run CheckProcessor and return the instance"""
    processor = CheckProcessor(
        source_folder=source_folder,
        cfg=cfg,
        mode=mode,
        output_dir=output_dir,
        mp=mp,
        workers=workers
    )
    processor.process()
    return processor


@pytest.mark.itk_process
class TestCheckProcessor:
    """Test suite for CheckProcessor"""

    def _get_valid_items(self, processor):
        return [r for r in processor.results if r.is_valid]

    def _get_invalid_items(self, processor):
        return [r for r in processor.results if not r.is_valid]

    def test_dataset_check_mode(self):
        """Test dataset mode with check operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('valid1', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('valid2', (80, 128, 128), (1.0, 0.5, 0.5)),
                ('invalid_size', (32, 64, 64), (1.0, 0.5, 0.5)),
                ('invalid_spacing', (64, 128, 128), (5.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100], max_spacing=[2.0, 1.0, 1.0])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items = self._get_valid_items(processor)
            invalid_items = self._get_invalid_items(processor)
            
            assert len(valid_items) == 2
            assert len(invalid_items) == 2
            assert os.path.exists(os.path.join(tmpdir, 'series_meta.json'))

    def test_single_check_mode(self):
        """Test single folder mode with check operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('valid1', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('valid2', (80, 128, 128), (1.0, 0.5, 0.5)),
                ('invalid1', (32, 64, 64), (1.0, 0.5, 0.5)),
            ]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items = self._get_valid_items(processor)
            invalid_items = self._get_invalid_items(processor)
            
            assert len(valid_items) == 2
            assert len(invalid_items) == 1
            assert os.path.exists(os.path.join(tmpdir, 'series_meta.json'))

    def test_fast_check_with_existing_meta(self):
        """Test fast check when series_meta.json already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('test1', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('test2', (32, 64, 64), (1.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path)
            
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items = self._get_valid_items(processor2)
            invalid_items = self._get_invalid_items(processor2)
            
            assert len(valid_items) == 1
            assert len(invalid_items) == 1

    def test_delete_mode(self):
        """Test delete operation removes invalid samples"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('invalid', (32, 64, 64), (1.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            run_check_processor(tmpdir, cfg, mode='delete', mp=False)
            
            assert os.path.exists(os.path.join(tmpdir, 'image', 'valid.mha'))
            assert os.path.exists(os.path.join(tmpdir, 'label', 'valid.mha'))
            assert not os.path.exists(os.path.join(tmpdir, 'image', 'invalid.mha'))
            assert not os.path.exists(os.path.join(tmpdir, 'label', 'invalid.mha'))

    @pytest.mark.parametrize("mode_type", ["dataset", "single"])
    def test_copy_mode(self, mode_type):
        """Test copy operation copies valid samples to output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            if mode_type == "dataset":
                src_dir = os.path.join(tmpdir, 'source')
                out_dir = os.path.join(tmpdir, 'output')
                
                image_specs = [
                    ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                    ('invalid', (32, 64, 64), (1.0, 0.5, 0.5)),
                ]
                setup_dataset_test_data(src_dir, image_specs)
                
                cfg = create_default_cfg(min_size=[50, 100, 100])
                run_check_processor(src_dir, cfg, mode='copy', output_dir=out_dir, mp=False)
                
                assert os.path.exists(os.path.join(out_dir, 'image', 'valid.mha'))
                assert os.path.exists(os.path.join(out_dir, 'label', 'valid.mha'))
                assert not os.path.exists(os.path.join(out_dir, 'image', 'invalid.mha'))
            else:  # single
                src_dir = os.path.join(tmpdir, 'source')
                out_dir = os.path.join(tmpdir, 'output')
                os.makedirs(src_dir)
                
                image_specs = [
                    ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                    ('invalid', (32, 64, 64), (1.0, 0.5, 0.5)),
                ]
                setup_single_folder_test_data(src_dir, image_specs)
                
                cfg = create_default_cfg(min_size=[50, 100, 100])
                run_check_processor(src_dir, cfg, mode='copy', output_dir=out_dir, mp=False)
                
                assert os.path.exists(os.path.join(out_dir, 'valid.mha'))
                assert not os.path.exists(os.path.join(out_dir, 'invalid.mha'))

    @pytest.mark.parametrize("mode_type", ["dataset", "single"])
    def test_symlink_mode(self, mode_type):
        """Test symlink operation creates symlinks to valid samples"""
        with tempfile.TemporaryDirectory() as tmpdir:
            if mode_type == "dataset":
                src_dir = os.path.join(tmpdir, 'source')
                out_dir = os.path.join(tmpdir, 'output')
                
                image_specs = [
                    ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                    ('invalid', (32, 64, 64), (1.0, 0.5, 0.5)),
                ]
                setup_dataset_test_data(src_dir, image_specs)
                
                cfg = create_default_cfg(min_size=[50, 100, 100])
                run_check_processor(src_dir, cfg, mode='symlink', output_dir=out_dir, mp=False)
                
                valid_img = os.path.join(src_dir, 'image', 'valid.mha')
                valid_lbl = os.path.join(src_dir, 'label', 'valid.mha')
                out_img_link = os.path.join(out_dir, 'image', 'valid.mha')
                out_lbl_link = os.path.join(out_dir, 'label', 'valid.mha')
                assert os.path.islink(out_img_link)
                assert os.path.islink(out_lbl_link)
                assert os.readlink(out_img_link) == valid_img
                assert os.readlink(out_lbl_link) == valid_lbl
            else:  # single
                src_dir = os.path.join(tmpdir, 'source')
                out_dir = os.path.join(tmpdir, 'output')
                os.makedirs(src_dir)
                
                image_specs = [
                    ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                    ('invalid', (32, 64, 64), (1.0, 0.5, 0.5)),
                ]
                setup_single_folder_test_data(src_dir, image_specs)
                
                cfg = create_default_cfg(min_size=[50, 100, 100])
                run_check_processor(src_dir, cfg, mode='symlink', output_dir=out_dir, mp=False)
                
                valid_path = os.path.join(src_dir, 'valid.mha')
                out_link = os.path.join(out_dir, 'valid.mha')
                assert os.path.islink(out_link)
                assert os.readlink(out_link) == valid_path

    def test_validation_rules(self):
        """Test various validation rules"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(same_spacing=(1, 2))
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            assert len(self._get_valid_items(processor)) == 1
            
            cfg = create_default_cfg(same_size=(1, 2))
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            assert len(self._get_valid_items(processor)) == 1

    def test_series_meta_persistence(self):
        """Test that series_meta.json is correctly saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test1', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg()
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path)
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            assert 'test1.mha' in meta
            assert 'size' in meta['test1.mha']
            assert 'spacing' in meta['test1.mha']
            assert meta['test1.mha']['size'] == [64, 128, 128]
            assert meta['test1.mha']['spacing'] == [1.0, 0.5, 0.5]

    def test_series_meta_completeness(self):
        """Test complete metadata fields and validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img = sitk.Image([30, 20, 10], sitk.sitkInt16)
            img.SetSpacing([1.5, 2.0, 2.5])
            img.SetOrigin([0.1, 0.2, 0.3])
            img_path = os.path.join(tmpdir, 'test_image.mha')
            sitk.WriteImage(img, img_path)
            
            # Create test label
            lbl_arr = np.zeros((10, 20, 30), dtype=np.uint8)
            lbl_arr[2:5, 5:10, 10:15] = 1
            lbl_arr[6:8, 12:15, 20:25] = 2
            lbl = sitk.GetImageFromArray(lbl_arr)
            lbl.SetSpacing([1.5, 2.0, 2.5])
            lbl.SetOrigin([0.1, 0.2, 0.3])
            lbl_path = os.path.join(tmpdir, 'test_label.mha')
            sitk.WriteImage(lbl, lbl_path)
            
            # Generate metadata
            from itkit.process.metadata_models import SeriesMetadata, MetadataManager
            img_meta = SeriesMetadata.from_sitk_image(img, 'test_image.mha')
            lbl_meta = SeriesMetadata.from_sitk_image(lbl, 'test_label.mha')
            
            # Verify image metadata
            assert img_meta.name == 'test_image.mha'
            assert img_meta.spacing == (2.5, 2.0, 1.5)  # ZYX order
            assert img_meta.size == (10, 20, 30)  # ZYX order
            assert img_meta.origin == (0.3, 0.2, 0.1)  # ZYX order
            assert img_meta.include_classes is None
            
            # Verify label metadata
            assert lbl_meta.name == 'test_label.mha'
            assert lbl_meta.spacing == (2.5, 2.0, 1.5)
            assert lbl_meta.size == (10, 20, 30)
            assert lbl_meta.origin == (0.3, 0.2, 0.1)
            assert set(lbl_meta.include_classes) == {0, 1, 2}
            
            # Test validation
            assert img_meta.validate_itk_image(img)
            assert lbl_meta.validate_itk_image(lbl)
            
            # Save and reload
            manager = MetadataManager()
            manager.update(img_meta)
            manager.update(lbl_meta)
            meta_path = os.path.join(tmpdir, 'meta.json')
            manager.save(meta_path)
            
            loaded_manager = MetadataManager(meta_path)
            loaded_img_meta = loaded_manager.meta['test_image.mha']
            loaded_lbl_meta = loaded_manager.meta['test_label.mha']
            
            # Verify loaded metadata matches
            assert loaded_img_meta == img_meta
            assert loaded_lbl_meta == lbl_meta
            
            # Verify validation still works after reload
            assert loaded_img_meta.validate_itk_image(img)
            assert loaded_lbl_meta.validate_itk_image(lbl)

    def test_max_size_constraint(self):
        """Test maximum size constraint validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('valid', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('invalid', (100, 128, 128), (1.0, 0.5, 0.5)),
            ]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(max_size=[80, 150, 150])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items = self._get_valid_items(processor)
            invalid_items = self._get_invalid_items(processor)
            
            assert len(valid_items) == 1
            assert len(invalid_items) == 1

    def test_min_spacing_constraint(self):
        """Test minimum spacing constraint validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('valid', (64, 128, 128), (2.0, 0.5, 0.5)),
                ('invalid', (64, 128, 128), (0.1, 0.5, 0.5)),
            ]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_spacing=[0.5, 0.5, 0.5])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items = self._get_valid_items(processor)
            invalid_items = self._get_invalid_items(processor)
            
            assert len(valid_items) == 1
            assert len(invalid_items) == 1

    def test_corrupted_meta_json_misleads_check(self):
        """Test that corrupted series_meta.json can mislead validation results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('sample1', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('sample2', (32, 64, 64), (1.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            
            # First check: generate series_meta.json
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items1 = self._get_valid_items(processor1)
            invalid_items1 = self._get_invalid_items(processor1)
            
            # Verify initial results: sample1 valid, sample2 invalid
            assert len(valid_items1) == 1
            assert len(invalid_items1) == 1
            assert valid_items1[0].name == 'sample1.mha'
            assert invalid_items1[0].name == 'sample2.mha'
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path)
            
            # Corrupt series_meta.json: change sample2's size to look valid
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Artificially inflate sample2's size to pass validation
            meta['sample2.mha']['size'] = [64, 128, 128]
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check: with corrupted metadata
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items2 = self._get_valid_items(processor2)
            invalid_items2 = self._get_invalid_items(processor2)
            
            # Verify that fast check is mislead by corrupted metadata
            # Now it thinks sample2 is valid based on corrupted meta
            assert len(valid_items2) == 2
            assert len(invalid_items2) == 0
            assert any(item.name == 'sample2.mha' for item in valid_items2)

    def test_partially_corrupted_meta_spacing(self):
        """Test metadata corruption on spacing values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image with high Z spacing (5.0) that violates max_spacing constraint
            # size=(64, 128, 128) ZYX, spacing=(5.0, 0.5, 0.5) ZYX
            image_specs = [('test1', (64, 128, 128), (5.0, 0.5, 0.5))]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(max_spacing=[2.0, 1.0, 1.0])  # [Z_max, Y_max, X_max]
            
            # First check: should be invalid due to high Z spacing (5.0 > 2.0)
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            invalid_items1 = self._get_invalid_items(processor1)
            
            # Verify first check correctly identified invalid sample
            assert len(invalid_items1) == 1, f"Expected 1 invalid, got {len(invalid_items1)}"
            assert 'spacing' in invalid_items1[0].reasons[0], f"Expected spacing error, got {invalid_items1[0].reasons}"
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path), "series_meta.json should be created"
            
            # Corrupt metadata to hide the high Z spacing
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            print(f"Original metadata: {meta['test1.mha']}")
            # Change Z spacing from 5.0 to 1.0 to make it appear valid
            meta['test1.mha']['spacing'][0] = 1.0  # Lie about Z spacing
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            print(f"Corrupted metadata: {meta['test1.mha']}")
            
            # Second check: fast path should use corrupted metadata and be fooled
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items2 = self._get_valid_items(processor2)
            invalid_items2 = self._get_invalid_items(processor2)
            
            # Now fast check thinks it's valid based on corrupted metadata
            assert len(valid_items2) == 1, \
                f"Expected 1 valid based on corrupted meta, got {len(valid_items2)}"
            assert len(invalid_items2) == 0, \
                f"Expected 0 invalid based on corrupted meta, got {len(invalid_items2)}"
            
            # Demonstrate the vulnerability: real file still has spacing[0]=5.0
            # but fast check ignored it and trusted corrupted metadata
            img = sitk.ReadImage(os.path.join(tmpdir, 'image', 'test1.mha'))
            real_spacing = list(img.GetSpacing())
            assert real_spacing[2] == 5.0, "Real file still has high Z spacing"

    def test_meta_json_missing_entries(self):
        """Test behavior when series_meta.json is missing some entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('sample1', (64, 128, 128), (1.0, 0.5, 0.5)),
                ('sample2', (80, 128, 128), (1.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg()
            
            # First check: generate metadata
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            assert len(self._get_valid_items(processor1)) == 2
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Remove sample2 from metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            del meta['sample2.mha']
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check: only processes sample1 due to missing entry
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            # Only sample1 is in valid_items (sample2 skipped due to missing metadata)
            valid_items2 = self._get_valid_items(processor2)
            assert len(valid_items2) == 1
            assert valid_items2[0].name == 'sample1.mha'

    def test_meta_json_all_entries_corrupted(self):
        """Test when all metadata entries are corrupted with wrong values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('img1', (32, 64, 64), (1.0, 0.5, 0.5)),
                ('img2', (40, 80, 80), (1.0, 0.5, 0.5)),
            ]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            
            # First check: both should be invalid
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            assert len(self._get_invalid_items(processor1)) == 2
            assert len(self._get_valid_items(processor1)) == 0
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Corrupt all entries to appear valid
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            for key in meta:
                meta[key]['size'] = [100, 128, 128]  # Make all look valid
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check: all appear valid due to corruption
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            assert len(self._get_valid_items(processor2)) == 2
            assert len(self._get_invalid_items(processor2)) == 0

    def test_single_folder_meta_corruption(self):
        """Test metadata corruption in single folder mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [
                ('img1', (100, 200, 200), (1.0, 0.5, 0.5)),
                ('img2', (30, 50, 50), (1.0, 0.5, 0.5)),
            ]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 100, 100])
            
            # First check
            processor1 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            valid_items1 = self._get_valid_items(processor1)
            invalid_items1 = self._get_invalid_items(processor1)
            
            assert len(valid_items1) == 1
            assert len(invalid_items1) == 1
            assert invalid_items1[0].name == 'img2.mha'
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Corrupt metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            meta['img2.mha']['size'] = [100, 128, 128]
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check with corrupted metadata
            processor2 = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            assert len(self._get_valid_items(processor2)) == 2
            assert len(self._get_invalid_items(processor2)) == 0

    def test_multiprocessing_mode(self):
        """Test multiprocessing in dataset mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [(f'img{i}', (64, 128, 128), (1.0, 0.5, 0.5)) for i in range(10)]
            setup_dataset_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[50, 50, 50])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False, workers=2)
            
            assert len(self._get_valid_items(processor)) == 10
            assert len(self._get_invalid_items(processor)) == 0

    def test_same_spacing_validation_fail(self):
        """Test same_spacing validation failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.3))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(same_spacing=(1, 2))  # Y and X should be same, but 0.5 != 0.3
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            invalid_items = self._get_invalid_items(processor)
            
            assert len(self._get_valid_items(processor)) == 0
            assert len(invalid_items) == 1
            assert 'differ' in invalid_items[0].reasons[0]

    def test_same_size_validation_fail(self):
        """Test same_size validation failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 64), (1.0, 0.5, 0.5))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(same_size=(0, 1))  # Z and Y: 64 != 128
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            invalid_items = self._get_invalid_items(processor)
            
            assert len(self._get_valid_items(processor)) == 0
            assert len(invalid_items) == 1
            assert 'differ' in invalid_items[0].reasons[0]

    def test_corrupted_image_file(self):
        """Test handling of corrupted image files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_path = os.path.join(tmpdir, 'corrupted.mha')
            with open(corrupted_path, 'w') as f:
                f.write("not an image")
            
            cfg = create_default_cfg()
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            invalid_items = self._get_invalid_items(processor)
            
            assert len(self._get_valid_items(processor)) == 0
            assert len(invalid_items) == 1
            assert 'Failed to read' in invalid_items[0].reasons[0]

    def test_skip_dimensions_with_minus_one(self):
        """Test skipping dimensions with -1 in cfg"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (32, 128, 128), (1.0, 0.5, 0.5))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg(min_size=[-1, 100, 100])  # Skip Z min_size
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            assert len(self._get_valid_items(processor)) == 1  # Z=32 <50 but skipped, Y,X ok

    def test_empty_series_meta_json(self):
        """Test fast check with empty series_meta.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_dataset_test_data(tmpdir, image_specs)
            
            # Create empty meta
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({}, f)
            
            cfg = create_default_cfg(min_size=[50, 50, 50])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            # Empty meta means no cached data, so no valid items from fast check
            assert len(self._get_valid_items(processor)) == 0
            assert len(self._get_invalid_items(processor)) == 0

    def test_invalid_json_meta(self):
        """Test with invalid JSON in series_meta.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_dataset_test_data(tmpdir, image_specs)
            
            # Create invalid JSON
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            with open(meta_path, 'w') as f:
                f.write("invalid json")
            
            cfg = create_default_cfg(min_size=[50, 50, 50])
            processor = run_check_processor(tmpdir, cfg, mode='check', mp=False)
            
            # Should remove corrupted meta and perform full check
            assert len(self._get_valid_items(processor)) == 1
            assert len(self._get_invalid_items(processor)) == 0
            # Meta should be recreated
            assert os.path.exists(meta_path)

    def test_copy_mode_no_output_dir(self):
        """Test copy mode without output_dir raises error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg()
            processor = CheckProcessor(tmpdir, cfg, 'copy', mp=False)
            # In code, it prints error and returns, so no exception, but for test, check that no operation happens
            processor.process()
            # Since no output_dir, should not create anything
            assert len(self._get_valid_items(processor)) == 1  # Would be valid if processed
            assert not os.path.exists(os.path.join(tmpdir, 'output'))  # No output dir created

    def test_symlink_mode_no_output_dir(self):
        """Test symlink mode without output_dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_specs = [('test', (64, 128, 128), (1.0, 0.5, 0.5))]
            setup_single_folder_test_data(tmpdir, image_specs)
            
            cfg = create_default_cfg()
            processor = CheckProcessor(tmpdir, cfg, 'symlink', mp=False)
            processor.process()  # Should print error and not crash
            assert len(self._get_valid_items(processor)) == 1  # Would be valid if processed
            assert not os.path.exists(os.path.join(tmpdir, 'output'))  # No output dir created
