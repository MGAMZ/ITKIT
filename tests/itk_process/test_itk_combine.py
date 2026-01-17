import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from itkit.process.itk_combine import CombineProcessor, MappingRule, SourceSpec


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def _write_label(folder: str, name: str, array: np.ndarray) -> str:
    path = os.path.join(folder, name)
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, path, True)
    return path


def _make_array(shape=(3, 3, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


@pytest.mark.itk_process
class TestCombineProcessor:
    def test_two_sources_simple_mapping(self, temp_dir):
        src_a = os.path.join(temp_dir, "A")
        src_b = os.path.join(temp_dir, "B")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(src_a)
        os.makedirs(src_b)
        os.makedirs(out_dir)

        arr_a = _make_array()
        arr_a[0, :, :] = 1
        arr_b = _make_array()
        arr_b[1, :, :] = 1

        _write_label(src_a, "case.mha", arr_a)
        _write_label(src_b, "case.mha", arr_b)

        sources = [
            SourceSpec(name="A", folder=Path(src_a)),
            SourceSpec(name="B", folder=Path(src_b)),
        ]
        rules = [
            MappingRule(source_name="A", source_labels=(1,), target_label=1),
            MappingRule(source_name="B", source_labels=(1,), target_label=2),
        ]

        processor = CombineProcessor(sources=sources, dest_folder=Path(out_dir), mapping_rules=rules)
        processor.process("Combine")

        output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out_dir, "case.mha")))
        assert np.all(output[0, :, :] == 1)
        assert np.all(output[1, :, :] == 2)
        assert np.all(output[2, :, :] == 0)

    def test_two_sources_complex_mapping(self, temp_dir):
        src_a = os.path.join(temp_dir, "A")
        src_b = os.path.join(temp_dir, "B")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(src_a)
        os.makedirs(src_b)
        os.makedirs(out_dir)

        arr_a = _make_array()
        arr_a[0, :, :] = 1
        arr_a[1, :, :] = 2
        arr_b = _make_array()
        arr_b[0, :, :] = 1
        arr_b[2, :, :] = 2

        _write_label(src_a, "case.mha", arr_a)
        _write_label(src_b, "case.mha", arr_b)

        sources = [
            SourceSpec(name="A", folder=Path(src_a)),
            SourceSpec(name="B", folder=Path(src_b)),
        ]
        rules = [
            MappingRule(source_name="A", source_labels=(1,), target_label=1),
            MappingRule(source_name="A", source_labels=(2,), target_label=4),
            MappingRule(source_name="B", source_labels=(1,), target_label=2),
            MappingRule(source_name="B", source_labels=(2,), target_label=3),
        ]

        processor = CombineProcessor(sources=sources, dest_folder=Path(out_dir), mapping_rules=rules)
        processor.process("Combine")

        output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out_dir, "case.mha")))
        assert np.all(output[0, :, :] == 1)
        assert np.all(output[1, :, :] == 4)
        assert np.all(output[2, :, :] == 3)

    def test_three_sources_mapping(self, temp_dir):
        src_a = os.path.join(temp_dir, "A")
        src_b = os.path.join(temp_dir, "B")
        src_c = os.path.join(temp_dir, "C")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(src_a)
        os.makedirs(src_b)
        os.makedirs(src_c)
        os.makedirs(out_dir)

        arr_a = _make_array()
        arr_a[0, :, :] = 1
        arr_b = _make_array()
        arr_b[1, :, :] = 1
        arr_c = _make_array()
        arr_c[2, :, :] = 1

        _write_label(src_a, "case.mha", arr_a)
        _write_label(src_b, "case.mha", arr_b)
        _write_label(src_c, "case.mha", arr_c)

        sources = [
            SourceSpec(name="A", folder=Path(src_a)),
            SourceSpec(name="B", folder=Path(src_b)),
            SourceSpec(name="C", folder=Path(src_c)),
        ]
        rules = [
            MappingRule(source_name="A", source_labels=(1,), target_label=1),
            MappingRule(source_name="B", source_labels=(1,), target_label=2),
            MappingRule(source_name="C", source_labels=(1,), target_label=3),
        ]

        processor = CombineProcessor(sources=sources, dest_folder=Path(out_dir), mapping_rules=rules)
        processor.process("Combine")

        output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out_dir, "case.mha")))
        assert np.all(output[0, :, :] == 1)
        assert np.all(output[1, :, :] == 2)
        assert np.all(output[2, :, :] == 3)

    def test_four_sources_priority(self, temp_dir):
        src_a = os.path.join(temp_dir, "A")
        src_b = os.path.join(temp_dir, "B")
        src_c = os.path.join(temp_dir, "C")
        src_d = os.path.join(temp_dir, "D")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(src_a)
        os.makedirs(src_b)
        os.makedirs(src_c)
        os.makedirs(src_d)
        os.makedirs(out_dir)

        arr_a = _make_array()
        arr_a[0, :, :] = 1
        arr_b = _make_array()
        arr_b[0, :, :] = 1
        arr_c = _make_array()
        arr_c[1, :, :] = 2
        arr_d = _make_array()
        arr_d[1, :, :] = 2

        _write_label(src_a, "case.mha", arr_a)
        _write_label(src_b, "case.mha", arr_b)
        _write_label(src_c, "case.mha", arr_c)
        _write_label(src_d, "case.mha", arr_d)

        sources = [
            SourceSpec(name="A", folder=Path(src_a)),
            SourceSpec(name="B", folder=Path(src_b)),
            SourceSpec(name="C", folder=Path(src_c)),
            SourceSpec(name="D", folder=Path(src_d)),
        ]
        rules = [
            MappingRule(source_name="B", source_labels=(1,), target_label=5),
            MappingRule(source_name="A", source_labels=(1,), target_label=1),
            MappingRule(source_name="D", source_labels=(2,), target_label=6),
            MappingRule(source_name="C", source_labels=(2,), target_label=2),
        ]

        processor = CombineProcessor(sources=sources, dest_folder=Path(out_dir), mapping_rules=rules)
        processor.process("Combine")

        output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(out_dir, "case.mha")))
        assert np.all(output[0, :, :] == 5)
        assert np.all(output[1, :, :] == 6)
        assert np.all(output[2, :, :] == 0)
