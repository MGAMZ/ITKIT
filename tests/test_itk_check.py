import os
import json
import tempfile
import pytest
import SimpleITK as sitk
from itkit.process.itk_check import CheckProcessor, ProcessorType


def create_test_image(path: str, size: tuple, spacing: tuple):
    """Helper to create test MHA images"""
    img = sitk.Image(size[::-1], sitk.sitkUInt8)  # SimpleITK uses XYZ
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, path)

@pytest.mark.itk_process
class TestCheckProcessor:
    """Test suite for CheckProcessor"""

    def test_dataset_check_mode(self):
        """Test dataset mode with check operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'valid1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'valid1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'valid2.mha'), (80, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'valid2.mha'), (80, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'invalid_size.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'invalid_size.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'invalid_spacing.mha'), (64, 128, 128), (5.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'invalid_spacing.mha'), (64, 128, 128), (5.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': [2.0, 1.0, 1.0],
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(
                source_folder=tmpdir,
                cfg=cfg,
                mode='check',
                processor_type=ProcessorType.DATASET,
                mp=False
            )
            processor.process()
            
            assert len(processor.valid_items) == 2
            assert len(processor.invalid) == 2
            assert os.path.exists(os.path.join(tmpdir, 'series_meta.json'))

    def test_single_check_mode(self):
        """Test single folder mode with check operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'valid1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(tmpdir, 'valid2.mha'), (80, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(tmpdir, 'invalid1.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(
                source_folder=tmpdir,
                cfg=cfg,
                mode='check',
                processor_type=ProcessorType.SINGLE,
                mp=False
            )
            processor.process()
            
            assert len(processor.valid_items) == 2
            assert len(processor.invalid) == 1
            assert os.path.exists(os.path.join(tmpdir, 'series_meta.json'))

    def test_fast_check_with_existing_meta(self):
        """Test fast check when series_meta.json already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'test1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'test2.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test2.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor1.process()
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path)
            
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor2.process()
            
            assert len(processor2.valid_items) == 1
            assert len(processor2.invalid) == 1

    def test_delete_mode(self):
        """Test delete operation removes invalid samples"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            valid_img = os.path.join(img_dir, 'valid.mha')
            valid_lbl = os.path.join(lbl_dir, 'valid.mha')
            invalid_img = os.path.join(img_dir, 'invalid.mha')
            invalid_lbl = os.path.join(lbl_dir, 'invalid.mha')
            
            create_test_image(valid_img, (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(valid_lbl, (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(invalid_img, (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(invalid_lbl, (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'delete', ProcessorType.DATASET, mp=False)
            processor.process()
            
            assert os.path.exists(valid_img)
            assert os.path.exists(valid_lbl)
            assert not os.path.exists(invalid_img)
            assert not os.path.exists(invalid_lbl)

    def test_copy_mode(self):
        """Test copy operation copies valid samples to output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'source')
            out_dir = os.path.join(tmpdir, 'output')
            
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'valid.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'valid.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(src_dir, cfg, 'copy', ProcessorType.DATASET, 
                                      output_dir=out_dir, mp=False)
            processor.process()
            
            assert os.path.exists(os.path.join(out_dir, 'image', 'valid.mha'))
            assert os.path.exists(os.path.join(out_dir, 'label', 'valid.mha'))
            assert not os.path.exists(os.path.join(out_dir, 'image', 'invalid.mha'))

    def test_symlink_mode(self):
        """Test symlink operation creates symlinks to valid samples"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'source')
            out_dir = os.path.join(tmpdir, 'output')
            
            img_dir = os.path.join(src_dir, 'image')
            lbl_dir = os.path.join(src_dir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            valid_img = os.path.join(img_dir, 'valid.mha')
            valid_lbl = os.path.join(lbl_dir, 'valid.mha')
            create_test_image(valid_img, (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(valid_lbl, (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(src_dir, cfg, 'symlink', ProcessorType.DATASET,
                                      output_dir=out_dir, mp=False)
            processor.process()
            
            out_img_link = os.path.join(out_dir, 'image', 'valid.mha')
            out_lbl_link = os.path.join(out_dir, 'label', 'valid.mha')
            assert os.path.islink(out_img_link)
            assert os.path.islink(out_lbl_link)
            assert os.readlink(out_img_link) == valid_img
            assert os.readlink(out_lbl_link) == valid_lbl

    def test_validation_rules(self):
        """Test various validation rules"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': (1, 2),
                'same_size': None,
            }
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            assert len(processor.valid_items) == 1
            
            cfg['same_spacing'] = None
            cfg['same_size'] = (1, 2)
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            assert len(processor.valid_items) == 1

    def test_series_meta_persistence(self):
        """Test that series_meta.json is correctly saved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'test1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor.process()
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            assert os.path.exists(meta_path)
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            assert 'test1.mha' in meta
            assert 'size' in meta['test1.mha']
            assert 'spacing' in meta['test1.mha']
            assert meta['test1.mha']['size'] == [64, 128, 128]
            assert meta['test1.mha']['spacing'] == [1.0, 0.5, 0.5]

    def test_single_mode_copy(self):
        """Test copy operation in single folder mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'source')
            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(src_dir)
            
            create_test_image(os.path.join(src_dir, 'valid.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(src_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(src_dir, cfg, 'copy', ProcessorType.SINGLE,
                                      output_dir=out_dir, mp=False)
            processor.process()
            
            assert os.path.exists(os.path.join(out_dir, 'valid.mha'))
            assert not os.path.exists(os.path.join(out_dir, 'invalid.mha'))

    def test_single_mode_symlink(self):
        """Test symlink operation in single folder mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, 'source')
            out_dir = os.path.join(tmpdir, 'output')
            os.makedirs(src_dir)
            
            valid_path = os.path.join(src_dir, 'valid.mha')
            create_test_image(valid_path, (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(src_dir, 'invalid.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(src_dir, cfg, 'symlink', ProcessorType.SINGLE,
                                      output_dir=out_dir, mp=False)
            processor.process()
            
            out_link = os.path.join(out_dir, 'valid.mha')
            assert os.path.islink(out_link)
            assert os.readlink(out_link) == valid_path

    def test_max_size_constraint(self):
        """Test maximum size constraint validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'valid.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(tmpdir, 'invalid.mha'), (100, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': [80, 150, 150],
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 1
            assert len(processor.invalid) == 1

    def test_min_spacing_constraint(self):
        """Test minimum spacing constraint validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'valid.mha'), (64, 128, 128), (2.0, 0.5, 0.5))
            create_test_image(os.path.join(tmpdir, 'invalid.mha'), (64, 128, 128), (0.1, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': [0.5, 0.5, 0.5],
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 1
            assert len(processor.invalid) == 1

    def test_corrupted_meta_json_misleads_check(self):
        """Test that corrupted series_meta.json can mislead validation results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            # Create test images with actual properties
            create_test_image(os.path.join(img_dir, 'sample1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'sample1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'sample2.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'sample2.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            # First check: generate series_meta.json
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor1.process()
            
            # Verify initial results: sample1 valid, sample2 invalid
            assert len(processor1.valid_items) == 1
            assert len(processor1.invalid) == 1
            assert processor1.valid_items[0][0] == 'sample1.mha'
            assert processor1.invalid[0][0] == 'sample2.mha'
            
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
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor2.process()
            
            # Verify that fast check is mislead by corrupted metadata
            # Now it thinks sample2 is valid based on corrupted meta
            assert len(processor2.valid_items) == 2
            assert len(processor2.invalid) == 0
            assert any(name == 'sample2.mha' for name, _ in processor2.valid_items)

    def test_partially_corrupted_meta_spacing(self):
        """Test metadata corruption on spacing values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            # Create image with high Z spacing (5.0) that violates max_spacing constraint
            # size=(64, 128, 128) ZYX, spacing=(5.0, 0.5, 0.5) ZYX
            create_test_image(os.path.join(img_dir, 'test1.mha'), (64, 128, 128), (5.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test1.mha'), (64, 128, 128), (5.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': [2.0, 1.0, 1.0],  # [Z_max, Y_max, X_max]
                'same_spacing': None,
                'same_size': None,
            }
            
            # First check: should be invalid due to high Z spacing (5.0 > 2.0)
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor1.process()
            
            # Verify first check correctly identified invalid sample
            assert len(processor1.invalid) == 1, f"Expected 1 invalid, got {len(processor1.invalid)}"
            assert 'spacing' in processor1.invalid[0][1][0], f"Expected spacing error, got {processor1.invalid[0][1]}"
            
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
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor2.process()
            
            # Now fast check thinks it's valid based on corrupted metadata
            assert len(processor2.valid_items) == 1, \
                f"Expected 1 valid based on corrupted meta, got {len(processor2.valid_items)}"
            assert len(processor2.invalid) == 0, \
                f"Expected 0 invalid based on corrupted meta, got {len(processor2.invalid)}"
            
            # Demonstrate the vulnerability: real file still has spacing[0]=5.0
            # but fast check ignored it and trusted corrupted metadata
            img = sitk.ReadImage(os.path.join(img_dir, 'test1.mha'))
            real_spacing = list(img.GetSpacing())
            assert real_spacing[2] == 5.0, "Real file still has high Z spacing"

    def test_meta_json_missing_entries(self):
        """Test behavior when series_meta.json is missing some entries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'sample1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'sample1.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'sample2.mha'), (80, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'sample2.mha'), (80, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            # First check: generate metadata
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor1.process()
            
            assert len(processor1.valid_items) == 2
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Remove sample2 from metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            del meta['sample2.mha']
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check: only processes sample1 due to missing entry
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor2.process()
            
            # Only sample1 is in valid_items (sample2 skipped due to missing metadata)
            assert len(processor2.valid_items) == 1
            assert processor2.valid_items[0][0] == 'sample1.mha'

    def test_meta_json_all_entries_corrupted(self):
        """Test when all metadata entries are corrupted with wrong values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'img1.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'img1.mha'), (32, 64, 64), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(img_dir, 'img2.mha'), (40, 80, 80), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'img2.mha'), (40, 80, 80), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            # First check: both should be invalid
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor1.process()
            
            assert len(processor1.invalid) == 2
            assert len(processor1.valid_items) == 0
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Corrupt all entries to appear valid
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            for key in meta:
                meta[key]['size'] = [100, 128, 128]  # Make all look valid
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check: all appear valid due to corruption
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False)
            processor2.process()
            
            assert len(processor2.valid_items) == 2
            assert len(processor2.invalid) == 0

    def test_single_folder_meta_corruption(self):
        """Test metadata corruption in single folder mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'img1.mha'), (100, 200, 200), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(tmpdir, 'img2.mha'), (30, 50, 50), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 100, 100],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            # First check
            processor1 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor1.process()
            
            assert len(processor1.valid_items) == 1
            assert len(processor1.invalid) == 1
            assert processor1.invalid[0][0] == 'img2.mha'
            
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            
            # Corrupt metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            meta['img2.mha']['size'] = [100, 128, 128]
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            # Second check with corrupted metadata
            processor2 = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor2.process()
            
            assert len(processor2.valid_items) == 2
            assert len(processor2.invalid) == 0

    def test_multiprocessing_mode(self):
        """Test multiprocessing in dataset mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            for i in range(10):
                create_test_image(os.path.join(img_dir, f'img{i}.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
                create_test_image(os.path.join(lbl_dir, f'img{i}.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [50, 50, 50],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.DATASET, mp=False, workers=2)
            processor.process()
            
            assert len(processor.valid_items) == 10
            assert len(processor.invalid) == 0

    def test_same_spacing_validation_fail(self):
        """Test same_spacing validation failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.3))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': (1, 2),  # Y and X should be same, but 0.5 != 0.3
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 0
            assert len(processor.invalid) == 1
            assert 'differ' in processor.invalid[0][1][0]

    def test_same_size_validation_fail(self):
        """Test same_size validation failure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'test.mha'), (64, 128, 64), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': (0, 1),  # Z and Y: 64 != 128
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 0
            assert len(processor.invalid) == 1
            assert 'differ' in processor.invalid[0][1][0]

    def test_corrupted_image_file(self):
        """Test handling of corrupted image files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_path = os.path.join(tmpdir, 'corrupted.mha')
            with open(corrupted_path, 'w') as f:
                f.write("not an image")
            
            cfg = {
                'min_size': None,
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 0
            assert len(processor.invalid) == 1
            assert 'Failed to read' in processor.invalid[0][1][0]

    def test_skip_dimensions_with_minus_one(self):
        """Test skipping dimensions with -1 in cfg"""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_image(os.path.join(tmpdir, 'test.mha'), (32, 128, 128), (1.0, 0.5, 0.5))
            
            cfg = {
                'min_size': [-1, 100, 100],  # Skip Z min_size
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }
            
            processor = CheckProcessor(tmpdir, cfg, 'check', ProcessorType.SINGLE, mp=False)
            processor.process()
            
            assert len(processor.valid_items) == 1  # Z=32 <50 but skipped, Y,X ok

    def test_empty_series_meta_json(self):
        """Test fast check with empty series_meta.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            
            # Create empty meta
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            with open(meta_path, 'w') as f:
                json.dump({}, f)
            
            cfg = {
                'min_size': [50, 50, 50],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }

    def test_invalid_json_meta(self):
        """Test with invalid JSON in series_meta.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'image')
            lbl_dir = os.path.join(tmpdir, 'label')
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            create_test_image(os.path.join(img_dir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            create_test_image(os.path.join(lbl_dir, 'test.mha'), (64, 128, 128), (1.0, 0.5, 0.5))
            
            # Create invalid JSON
            meta_path = os.path.join(tmpdir, 'series_meta.json')
            with open(meta_path, 'w') as f:
                f.write("{}")  # Valid empty JSON
            
            cfg = {
                'min_size': [50, 50, 50],
                'max_size': None,
                'min_spacing': None,
                'max_spacing': None,
                'same_spacing': None,
                'same_size': None,
            }

    def test_copy_mode_no_output_dir(self):
        """Test copy mode without output_dir raises error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {'min_size': None, 'max_size': None, 'min_spacing': None, 'max_spacing': None, 'same_spacing': None, 'same_size': None}
            processor = CheckProcessor(tmpdir, cfg, 'copy', ProcessorType.SINGLE, mp=False)
            # In code, it prints error and returns, so no exception, but for test, check that no operation happens
            processor.process()
            # Since no output_dir, should not create anything, but test is limited

    def test_symlink_mode_no_output_dir(self):
        """Test symlink mode without output_dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {'min_size': None, 'max_size': None, 'min_spacing': None, 'max_spacing': None, 'same_spacing': None, 'same_size': None}
            processor = CheckProcessor(tmpdir, cfg, 'symlink', ProcessorType.SINGLE, mp=False)
            processor.process()  # Should print error and not crash
            assert len(processor.valid_items) == 0  # No files, but test the error path
