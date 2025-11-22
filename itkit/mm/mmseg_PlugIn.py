import warnings
from collections import OrderedDict

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.structures import PixelData
from mmseg.evaluation.metrics import IoUMetric
from mmseg.structures import SegDataSample
from monai.metrics.confusion_matrix import (
    compute_confusion_matrix_metric,
    get_confusion_matrix,
)
from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
from prettytable import PrettyTable
from skimage.exposure import equalize_hist


class HistogramEqualization(BaseTransform):
    def __init__(self, image_size: tuple, ratio: float):
        assert image_size[0] == image_size[1], "Only support square shape for now."
        assert ratio < 1, "RoI out of bounds"
        self.RoI = self.create_circle_in_square(image_size[0], image_size[0] * ratio)
        self.nbins = image_size[0]

    @staticmethod
    def create_circle_in_square(size: int, radius: int) -> np.ndarray:
        # Create a square ndarray filled with zeros
        square = np.zeros((size, size))
        # Compute the coordinates of the center point
        center = size // 2
        # Compute the distance of each element to the center
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        # Set elements within radius to 1
        square[mask] = 1
        return square

    def RoI_HistEqual(self, image: np.ndarray):
        dtype_range = np.iinfo(image)
        normed_image = equalize_hist(image, nbins=self.nbins, mask=self.RoI)
        normed_image = (normed_image * dtype_range.max).astype(image.dtype)
        return normed_image

    def transform(self, results: dict) -> dict:
        assert isinstance(results["img"], list)
        for i, image in enumerate(results["img"]):
            results["img"][i] = self.RoI_HistEqual(image)
        return results


class IoUMetric_PerClass(IoUMetric):
    def __init__(self, iou_metrics: list[str]=['mIoU', 'mDice', 'mFscore'], *args, **kwargs):
        super().__init__(iou_metrics=iou_metrics, *args, **kwargs)

    def compute_metrics(self, results: list) -> dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect: torch.Tensor = sum(results[0])
        total_area_union: torch.Tensor = sum(results[1])
        total_area_pred_label: torch.Tensor = sum(results[2])
        total_area_label: torch.Tensor = sum(results[3])

        per_class_eval_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        # class averaged table
        class_avged_metrics = OrderedDict(
            {
                criterion: np.round(np.nanmean(criterion_value) * 100, 2)
                for criterion, criterion_value in per_class_eval_metrics.items()
            }
        )
        metrics = {}
        for key, val in class_avged_metrics.items():
            if key == "aAcc":
                metrics[key] = val
            else:
                metrics["m" + key] = val

        # each class table
        per_class_eval_metrics.pop("aAcc", None)
        per_classes_formatted_dict = OrderedDict(
            {
                criterion: [format(v, ".2f") for v in criterion_value * 100]
                for criterion, criterion_value in per_class_eval_metrics.items()
            }
        )
        per_classes_formatted_dict.update({"Class": self.dataset_meta["classes"]}) # type: ignore
        per_classes_formatted_dict.move_to_end("Class", last=False)
        terminal_table = PrettyTable()
        for key, val in per_classes_formatted_dict.items():
            terminal_table.add_column(key, val)

        # provide per class results for logger hook
        metrics["PerClass"] = per_classes_formatted_dict

        print_log("per class results:", 'current')
        print_log("\n" + terminal_table.get_string(), logger='current')

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.Tensor, label: torch.Tensor,
                            num_classes: int, ignore_index: int):
        pred_label = pred_label.to(device='cpu', dtype=torch.uint8)
        label = label.to(device='cpu', dtype=torch.uint8)
        return IoUMetric.intersect_and_union(pred_label, label, num_classes, ignore_index)


class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = {}
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 2:
                data = to_tensor(results["gt_seg_map"][None, ...])
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"])
            data_sample.gt_sem_seg = PixelData(data=data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


class MonaiSegMetrics(BaseMetric):
    """
    A metric evaluator that leverages MONAI's metric computation algorithms
    to compute comprehensive segmentation metrics including Dice, IoU, Recall, and Precision.

    This class follows the BaseMetric interface from mmengine:
    - process(): Computes per-sample metrics and stores in self.results
    - compute_metrics(): Aggregates collected results from all ranks

    Args:
        ignore_index (int): Index that will be ignored in evaluation. Default: 255.
        include_background (bool): Whether to include the background class in metrics. Default: True.
        num_classes (int, optional): Number of classes. If None, will be inferred from dataset_meta.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str, optional): Prefix for metric names.
    """

    def __init__(
        self,
        include_background: bool = True,
        num_classes: int | None = None,
        collect_device: str = 'cpu',
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.include_background = include_background
        self.num_classes = num_classes

    def process(self, data_batch: dict, data_samples: list) -> None:
        """
        Process one batch of data and data_samples.

        Computes per-sample metrics using MONAI's functions and stores serializable results in self.results.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (list): A batch of outputs from the model.
        """
        num_classes = self.num_classes or len(self.dataset_meta['classes'])

        for data_sample in data_samples:
            # Extract prediction and ground truth
            seg_logits = data_sample['seg_logits']['data']  # [C, Z, Y, X]
            pred_label = data_sample['pred_sem_seg']['data']  # [Z, Y, X]
            label = data_sample['gt_sem_seg']['data'].to(pred_label)  # [Z, Y, X]

            # Convert to one-hot format [1, C, Z, Y, X] as required by MONAI
            pred_onehot = self._to_onehot(pred_label, num_classes)
            label_onehot = self._to_onehot(label, num_classes)

            # Compute per-sample metrics using MONAI's official functions
            # compute_dice returns [B, C], we have `B=1` so squeeze to [C]
            dice_score = compute_dice(
                y_pred=pred_onehot,
                y=label_onehot,
                include_background=self.include_background,
                ignore_empty=True,
                num_classes=num_classes
            ).squeeze(0)  # [C]

            # compute_iou returns [B, C]
            iou_score = compute_iou(
                y_pred=pred_onehot,
                y=label_onehot,
                include_background=self.include_background,
                ignore_empty=True
            ).squeeze(0)  # [C]

            # get_confusion_matrix returns [B, C, 4] where last dim is [TP, FP, TN, FN]
            confusion_matrix = get_confusion_matrix(
                y_pred=pred_onehot,
                y=label_onehot,
                include_background=self.include_background
            ).squeeze(0)  # [C, 4]

            # Compute recall and precision from confusion matrix using MONAI's function
            # recall = TPR = TP / (TP + FN)
            recall_score = compute_confusion_matrix_metric("recall", confusion_matrix)  # [C]
            # precision = PPV = TP / (TP + FP)
            precision_score = compute_confusion_matrix_metric("precision", confusion_matrix)  # [C]

            # Store serializable tensors in self.results (required by BaseMetric)
            self.results.append({
                'dice': dice_score.cpu(),
                'iou': iou_score.cpu(),
                'recall': recall_score.cpu(),
                'precision': precision_score.cpu(),
            })

    def compute_metrics(self, results: list) -> dict:
        """
        Compute the metrics from processed results.

        Aggregates per-sample metrics collected from all processes.

        Args:
            results (list): The processed results collected from all ranks.

        Returns:
            dict: The computed metrics including mDice, mIoU, mRecall, mPrecision.
        """
        if not results:
            return {}

        # Stack all per-sample results
        all_dice = torch.stack([r['dice'] for r in results])  # [N, C]
        all_iou = torch.stack([r['iou'] for r in results])  # [N, C]
        all_recall = torch.stack([r['recall'] for r in results])  # [N, C]
        all_precision = torch.stack([r['precision'] for r in results])  # [N, C]

        # Compute mean across samples for each class
        dice_scores = torch.nanmean(all_dice, dim=0)  # [C]
        iou_scores = torch.nanmean(all_iou, dim=0)  # [C]
        recall_scores = torch.nanmean(all_recall, dim=0)  # [C]
        precision_scores = torch.nanmean(all_precision, dim=0)  # [C]

        # Compute mean across classes
        mean_dice = torch.nanmean(dice_scores).item() * 100
        mean_iou = torch.nanmean(iou_scores).item() * 100
        mean_recall = torch.nanmean(recall_scores).item() * 100
        mean_precision = torch.nanmean(precision_scores).item() * 100

        # Class averaged metrics (matching IoUMetric_PerClass format)
        class_avged_metrics = OrderedDict()
        class_avged_metrics['Dice'] = np.round(mean_dice, 2)
        class_avged_metrics['IoU'] = np.round(mean_iou, 2)
        class_avged_metrics['Recall'] = np.round(mean_recall, 2)
        class_avged_metrics['Precision'] = np.round(mean_precision, 2)

        metrics = {}
        for key, val in class_avged_metrics.items():
            metrics['m' + key] = val

        # Per-class results (matching IoUMetric_PerClass format)
        class_names = self.dataset_meta['classes']
        per_classes_formatted_dict = OrderedDict()
        per_classes_formatted_dict['Class'] = class_names
        per_classes_formatted_dict['Dice'] = [format(v.item() * 100, ".2f") for v in dice_scores]
        per_classes_formatted_dict['IoU'] = [format(v.item() * 100, ".2f") for v in iou_scores]
        per_classes_formatted_dict['Recall'] = [format(v.item() * 100, ".2f") for v in recall_scores]
        per_classes_formatted_dict['Precision'] = [format(v.item() * 100, ".2f") for v in precision_scores]

        # Create pretty table for terminal display
        terminal_table = PrettyTable()
        for key, val in per_classes_formatted_dict.items():
            terminal_table.add_column(key, val)

        # Provide per class results for logger hook
        metrics['PerClass'] = per_classes_formatted_dict

        print_log('per class results:', 'current')
        print_log('\n' + terminal_table.get_string(), logger='current')

        return metrics

    def _to_onehot(
        self, label_map: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert 3D label map to one-hot format.

        Args:
            label_map (torch.Tensor): Label map with shape [Z, Y, X].
            num_classes (int): Number of classes.
            ignore_index (int): Index to ignore.

        Returns:
            torch.Tensor: One-hot tensor with shape [1, C, Z, Y, X].
        """

        # Clip labels to valid range
        label_map_clipped = torch.clamp(label_map, 0, num_classes - 1)

        # Convert to one-hot: [Z, Y, X] -> [Z, Y, X, C]
        onehot = torch.nn.functional.one_hot(label_map_clipped.long(), num_classes)

        # Permute to channel-first: [Z, Y, X, C] -> [C, Z, Y, X] -> [1, C, Z, Y, X]
        return onehot.permute(3, 0, 1, 2).unsqueeze(0).to(torch.uint8)
