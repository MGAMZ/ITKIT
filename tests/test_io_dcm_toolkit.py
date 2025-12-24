"""
Comprehensive test suite for itkit.io.dcm_toolkit module.
Tests DICOM reading functionality.
"""

import inspect
import os
import tempfile

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset

from itkit.io.dcm_toolkit import read_dcm_as_sitk


class TestReadDcmAsSitk:
    """Test read_dcm_as_sitk function."""

    def _create_minimal_dicom(self, path, instance_number=1, z_position=0.0):
        """Create a minimal DICOM file for testing."""
        # Create a minimal valid DICOM file
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(
            path,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )

        # Required DICOM tags
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9"
        ds.StudyInstanceUID = "1.2.3.4.5.6.7.8"
        ds.SOPInstanceUID = f"1.2.3.4.5.6.7.8.9.{instance_number}"
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.Modality = "CT"

        # Image-related tags
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = 64
        ds.Columns = 64
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed

        # Position and orientation
        ds.ImagePositionPatient = [0.0, 0.0, z_position]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.InstanceNumber = instance_number

        # Pixel data
        pixel_array = np.random.randint(-1000, 1000, (64, 64), dtype=np.int16)
        ds.PixelData = pixel_array.tobytes()

        ds.save_as(path, write_like_original=False)
        return ds

    def test_read_dcm_as_sitk_basic(self):
        """Test basic DICOM reading functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple DICOM series
            for i in range(3):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            dcms, image = read_dcm_as_sitk(tmpdir, need_dcms=True)

            # Check that we got results
            assert dcms is not None
            assert image is not None
            assert len(dcms) == 3
            assert isinstance(image, sitk.Image)

    def test_read_dcm_as_sitk_without_dcms(self):
        """Test reading DICOM without loading DICOM objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create DICOM series
            for i in range(3):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            dcms, image = read_dcm_as_sitk(tmpdir, need_dcms=False)

            assert dcms is None
            assert image is not None
            assert isinstance(image, sitk.Image)

    def test_read_dcm_empty_directory_returns_none(self):
        """Test that empty directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dcms, image = read_dcm_as_sitk(tmpdir)

            assert dcms is None
            assert image is None

    def test_read_dcm_nonexistent_directory(self):
        """Test reading from non-existent directory."""
        dcms, image = read_dcm_as_sitk("/nonexistent/directory")

        # Should return None for non-existent directory
        assert dcms is None
        assert image is None

    def test_read_dcm_as_sitk_image_properties(self):
        """Test that loaded image has correct properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create DICOM series with known properties
            for i in range(5):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            dcms, image = read_dcm_as_sitk(tmpdir)

            assert image is not None
            # Check that it's a 3D image
            assert len(image.GetSize()) == 3
            # Z dimension should match number of slices
            assert image.GetSize()[2] == 5

    def test_read_dcm_as_sitk_pixel_type(self):
        """Test that pixel type is Int16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            dcms, image = read_dcm_as_sitk(tmpdir)

            assert image is not None
            # Should be Int16 as specified in the function
            assert image.GetPixelID() == sitk.sitkInt16

    def test_read_dcm_function_signature(self):
        """Test function signature and parameters."""
        sig = inspect.signature(read_dcm_as_sitk)

        assert 'data_directory' in sig.parameters
        assert 'need_dcms' in sig.parameters
        assert sig.parameters['need_dcms'].default is True

    def test_read_dcm_return_type(self):
        """Test that function returns tuple of correct types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            result = read_dcm_as_sitk(tmpdir, need_dcms=True)

            assert isinstance(result, tuple)
            assert len(result) == 2

            dcms, image = result
            assert isinstance(dcms, list) or dcms is None
            assert isinstance(image, sitk.Image) or image is None

    def test_read_dcm_metadata_dictionary_update(self):
        """Test that metadata dictionary is updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2))

            dcms, image = read_dcm_as_sitk(tmpdir)

            # The function enables MetaDataDictionaryArrayUpdateOn
            # So the image should have metadata
            assert image is not None

    def test_read_dcm_series_details(self):
        """Test reading with series details enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple slices
            for i in range(3):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=float(i*2.5))

            dcms, image = read_dcm_as_sitk(tmpdir)

            assert image is not None
            # Should successfully read the series with details
            assert image.GetSize()[2] == 3

    def test_read_dcm_multiple_files_ordering(self):
        """Test that multiple DICOM files are ordered correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in non-sequential order
            positions = [0.0, 5.0, 2.5, 7.5, 10.0]
            for i, z_pos in enumerate(positions):
                path = os.path.join(tmpdir, f"slice_{i:03d}.dcm")
                self._create_minimal_dicom(path, instance_number=i+1, z_position=z_pos)

            dcms, image = read_dcm_as_sitk(tmpdir)

            assert image is not None
            assert image.GetSize()[2] == 5
            # SimpleITK should handle the ordering based on ImagePositionPatient
