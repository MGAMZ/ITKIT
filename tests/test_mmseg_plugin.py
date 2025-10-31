import pytest
import torch
import numpy as np
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from itkit.mm.mmseg_PlugIn import MonaiSegMetrics


class TestMonaiSegMetrics:
    """Test suite for MonaiSegMetrics class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.num_classes = 3
        self.dataset_meta = {'classes': ['background', 'class1', 'class2']}
        self.metric = MonaiSegMetrics(
            include_background=True,
            num_classes=self.num_classes,
            collect_device='cpu',
            prefix=None
        )
        self.metric.dataset_meta = self.dataset_meta

    def create_mock_data_sample(self):
        """Create a mock data sample dict for testing."""
        # Create mock tensors
        seg_logits = torch.randn(3, 4, 8, 8)  # [C, Z, Y, X]
        pred_label = torch.randint(0, self.num_classes, (4, 8, 8))  # [Z, Y, X]
        gt_label = torch.randint(0, self.num_classes, (4, 8, 8))  # [Z, Y, X]

        # Create a simple dict to mimic SegDataSample
        data_sample = {
            'seg_logits': {'data': seg_logits},
            'pred_sem_seg': {'data': pred_label},
            'gt_sem_seg': {'data': gt_label}
        }

        return data_sample

    def test_process_single_sample(self):
        """Test processing a single data sample."""
        data_batch = {}  # Mock data batch
        data_samples = [self.create_mock_data_sample()]

        # Process the sample
        self.metric.process(data_batch, data_samples)

        # Check that results were added
        assert len(self.metric.results) == 1
        result = self.metric.results[0]

        # Check that all metrics are present
        assert 'dice' in result
        assert 'iou' in result
        assert 'recall' in result
        assert 'precision' in result

        # Check tensor shapes and types
        assert result['dice'].shape == (self.num_classes,)
        assert result['iou'].shape == (self.num_classes,)
        assert result['recall'].shape == (self.num_classes,)
        assert result['precision'].shape == (self.num_classes,)

        assert result['dice'].dtype == torch.float32
        assert result['iou'].dtype == torch.float32
        assert result['recall'].dtype == torch.float32
        assert result['precision'].dtype == torch.float32

    def test_process_multiple_samples(self):
        """Test processing multiple data samples."""
        data_batch = {}
        data_samples = [self.create_mock_data_sample() for _ in range(3)]

        self.metric.process(data_batch, data_samples)

        assert len(self.metric.results) == 3

    def test_compute_metrics(self):
        """Test computing aggregated metrics."""
        # Process some samples first
        data_batch = {}
        data_samples = [self.create_mock_data_sample() for _ in range(2)]
        self.metric.process(data_batch, data_samples)

        # Compute metrics
        results = self.metric.results
        metrics = self.metric.compute_metrics(results)

        # Check that aggregated metrics are present
        assert 'mDice' in metrics
        assert 'mIoU' in metrics
        assert 'mRecall' in metrics
        assert 'mPrecision' in metrics

        # Check that per-class results are present
        assert 'PerClass' in metrics
        per_class = metrics['PerClass']
        assert 'Class' in per_class
        assert 'Dice' in per_class
        assert 'IoU' in per_class
        assert 'Recall' in per_class
        assert 'Precision' in per_class

        # Check that class names match
        assert per_class['Class'] == self.dataset_meta['classes']

        # Check that per-class values are strings formatted to 2 decimals
        assert len(per_class['Dice']) == self.num_classes
        assert all(isinstance(v, str) for v in per_class['Dice'])

    def test_compute_metrics_empty_results(self):
        """Test computing metrics with empty results."""
        metrics = self.metric.compute_metrics([])
        assert metrics == {}

    def test_to_onehot(self):
        """Test the _to_onehot helper method."""
        label_map = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]])  # [Z=2, Y=2, X=2]
        onehot = self.metric._to_onehot(label_map, self.num_classes)

        # Check shape: [1, C, Z, Y, X]
        assert onehot.shape == (1, self.num_classes, 2, 2, 2)
        assert onehot.dtype == torch.uint8

        # Check that it's one-hot encoded
        # Sum across channel dimension should be 1
        assert torch.all(torch.sum(onehot, dim=1) == 1)

    def test_to_onehot_clamping(self):
        """Test that _to_onehot clamps invalid labels."""
        label_map = torch.tensor([[[-1, 1], [3, 0]]])  # Invalid labels: -1 and 3 (num_classes=3, valid 0-2)
        onehot = self.metric._to_onehot(label_map, self.num_classes)

        # Should clamp to valid range
        assert torch.all(onehot >= 0)
        assert torch.all(onehot <= 1)

    def test_perfect_and_worst_case_metrics(self):
        """Test metrics for perfect match and completely wrong predictions."""
        # Test perfect match: pred == gt
        perfect_pred = torch.ones(4, 8, 8, dtype=torch.long)  # All class 1
        perfect_gt = torch.ones(4, 8, 8, dtype=torch.long)    # All class 1
        
        perfect_data_sample = {
            'seg_logits': {'data': torch.randn(3, 4, 8, 8)},  # Dummy logits
            'pred_sem_seg': {'data': perfect_pred},
            'gt_sem_seg': {'data': perfect_gt}
        }
        
        self.metric.process({}, [perfect_data_sample])
        perfect_results = self.metric.compute_metrics(self.metric.results)
        
        # For perfect match, all metrics should be 100.0
        assert perfect_results['mDice'] == 100.0
        assert perfect_results['mIoU'] == 100.0
        assert perfect_results['mRecall'] == 100.0
        assert perfect_results['mPrecision'] == 100.0
        
        # Reset results for worst case
        self.metric.results = []
        
        # Test worst case: pred all background (0), gt all class 1
        worst_pred = torch.zeros(4, 8, 8, dtype=torch.long)  # All background
        worst_gt = torch.ones(4, 8, 8, dtype=torch.long)     # All class 1
        
        worst_data_sample = {
            'seg_logits': {'data': torch.randn(3, 4, 8, 8)},  # Dummy logits
            'pred_sem_seg': {'data': worst_pred},
            'gt_sem_seg': {'data': worst_gt}
        }
        
        self.metric.process({}, [worst_data_sample])
        worst_results = self.metric.compute_metrics(self.metric.results)
        
        # For completely wrong predictions, all metrics should be 0.0
        assert worst_results['mDice'] == 0.0
        assert worst_results['mIoU'] == 0.0
        assert worst_results['mRecall'] == 0.0
        assert worst_results['mPrecision'] == 0.0
