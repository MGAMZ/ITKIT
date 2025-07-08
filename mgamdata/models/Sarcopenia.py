import pdb
import warnings
from collections.abc import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from matplotlib.patches import Patch
from scipy.ndimage import zoom

from mmcv.transforms import BaseTransform
from mmengine.structures import BaseDataElement
from mmengine.evaluator.metric import BaseMetric
from mmengine.runner import Runner
from ..mm.mmseg_PlugIn import IoUMetric_PerClass
from ..mm.mmseg_Dev3D import Seg3DDataSample, VolumeData
from ..mm.visualization import BaseViser, BaseVisHook
from ..dataset.RenJi_Sarcopenia import CLASS_MAP, CLASS_MAP_AFTER_POSTSEG



class SeriesData(BaseDataElement):
    """Data structure for 1D annotations or predictions.

    所有存储在 ``data_fields`` 中的数据都满足以下要求：
    - 维度为 [C(可选), Z]。
    - 它们应当具有相同的 Z 维度。
    """

    def __setattr__(self, name: str, value: Tensor | np.ndarray):
        """Set attributes of ``VolumeData``.

        若传入数据为 1D（即形状仅 [Z]），则自动加一维成为 [1, Z]。

        Args:
            name (str): 设置属性的名称。
            value (Union[Tensor, np.ndarray]): 要存储的值，只能是
                `Tensor` 或 `np.ndarray`，形状必须满足 [C(可选), Z] 的要求。
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f"{name} 已被用作私有属性，无法再次修改。")
        else:
            assert isinstance(value, (Tensor, np.ndarray)), (
                f"无法设置 {type(value)}，仅支持 {(Tensor, np.ndarray)}。")

            if self.shape:
                # 只比较最后一维是否一致
                assert value.shape[-1] == self.shape[0], (
                    f"传入数据的 Z 维度 {value.shape[-1]} "
                    f"与当前 VolumeData 的 shape {self.shape} 不一致。"
                )

            # 检查维度必须为 1 或 2
            assert value.ndim in [1, 2], f"数据维度必须为 1 或 2，但当前为 {value.ndim}。"
            if value.ndim == 1:
                value = value[None]  # 在最前面加一维
                warnings.warn(f"已将输入从 (Z,) 自动转换为 (1, Z)。当前形状为 {value.shape}。")

            super().__setattr__(name, value)

    def __getitem__(self, item: int | slice) -> "VolumeData":
        """
        Args:
            item (Union[int, slice]): 根据下标或切片从 Z 维度获取数据。

        Returns:
            :obj:`VolumeData`: 切片后的数据。
        """
        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, (int, slice)):
            for k, v in self.items():
                setattr(new_data, k, v[:, item])
        else:
            raise TypeError(f"仅支持 int 或 slice 类型的索引，但传入 {type(item)}。")
        return new_data

    @property
    def shape(self):
        """返回当前数据的 Z 维度，比如 (100,)。忽略通道维度。"""
        if len(self._data_fields) > 0:
            # 只返回最后一维的大小
            return (self.values()[0].shape[-1],)
        else:
            return None


class SarcopeniaBatchAugmentor:
    def __init__(self, patch_size:tuple[int, int, int], num_patches_per_batch:int=2):
        self.patch_size = patch_size
        self.num_patches_per_batch = num_patches_per_batch

    def __call__(self, inputs:Tensor, data_samples:list[Seg3DDataSample]) -> tuple[Tensor, list[Seg3DDataSample]]:
        """Batch augmentation for 3D data.
        
        Args:
            inputs (Tensor): Input data, shape (N, C, Z, Y, X).
            data_samples (list[Seg3DDataSample]): List of data samples, num of elements is `N`.
                - gt_sem_seg (data:SeriesData): tensor (Z, Y, X)
        
        Returns
            inputs (Tensor): Augmented input data, shape(N * num_patches, C, patch_Z, patch_Y, patch_X).
            data_samples (list[Seg3DDataSample]): List of data samples, num of elements is `N * num_patches`.
                - gt_sem_seg (data:SeriesData): tensor (Z, Y, X)
        """
        import random, copy
        # 获取原始尺寸
        N, C, Z, Y, X = inputs.shape
        pz, py, px = self.patch_size
        assert pz <= Z and py <= Y and px <= X, \
            f"patch_size {self.patch_size} invalid for input size {(Z, Y, X)}"
        new_inputs = []
        new_samples = []
        # 对每个样本生成 num_patches_per_batch 个随机patch
        for i in range(N):
            vol = inputs[i]
            sample = data_samples[i]
            for _ in range(self.num_patches_per_batch):
                z1 = random.randint(0, Z - pz)
                y1 = random.randint(0, Y - py)
                x1 = random.randint(0, X - px)
                # 切取patch
                patch_vol = vol[:, z1:z1+pz, y1:y1+py, x1:x1+px]
                new_inputs.append(patch_vol)
                # 克隆data_sample并切取对应标签
                new_sample = copy.deepcopy(sample)
                # 语义分割切片
                if hasattr(new_sample, 'gt_sem_seg'):
                    seg = new_sample.gt_sem_seg.data
                    seg_patch = seg[:, z1:z1+pz, y1:y1+py, x1:x1+px] if seg.ndim == 4 else seg[z1:z1+pz, y1:y1+py, x1:x1+px]
                    new_sample.gt_sem_seg.data = seg_patch
                new_samples.append(new_sample)
        # 合并并返回
        new_inputs_tensor = torch.stack(new_inputs, dim=0)
        del inputs, data_samples
        return new_inputs_tensor, new_samples


class L3_Evaluator(BaseMetric):
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        acc = []
        for sample in data_samples:
            pred = sample.get("pred_L3")
            gt = sample.get("gt_L3")
            acc.append((pred == gt).float())
        self.results.extend(acc)
    
    def compute_metrics(self, results: list) -> dict:
        return {"Val/acc_L3": np.mean(results)}


class L3_VisHook(BaseVisHook):
    def __init__(self, interval:int=1, draw:bool=True):
        self.interval = interval
        self.draw = draw
    
    def after_val_iter(self,
                       runner:Runner,
                       batch_idx: int,
                       data_batch: list[BaseDataElement],
                       outputs: list[BaseDataElement]) -> None:
        """
        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if self.draw is False or batch_idx % self.interval != 0:
            return
        
        # 创建具有多个子图的图像，每个子图对应一个输出
        if len(outputs) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 2))
            axes = [axes]
        else:
            fig, axes = plt.subplots(len(outputs), 1, figsize=(10, len(outputs)*2))
        
        # 逐样本绘制图像
        for i, (output, ax) in enumerate(zip(outputs, axes)):
            pred = output.get_field("pred_L3")  # [Z]
            gt = output.get_field("gt_L3")  # [Z]
            
            # 确保转换为numpy数组以便visualization
            if not isinstance(pred, np.ndarray):
                pred = np.array(pred)
            if not isinstance(gt, np.ndarray):
                gt = np.array(gt)
                
            # 创建一个2行的数组，第一行是pred，第二行是gt
            display_data = np.vstack((pred, gt))
            
            # 使用imshow绘制数据，binary cmap会将0显示为白色，1显示为黑色
            im = ax.imshow(display_data, aspect='auto', cmap='binary', interpolation='none')
            
            # 添加y轴标签
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Pred', 'GT'])
            
            # 添加标题
            ax.set_title(f'Sample {i+1}')
            
            # 如果序列很长，可以简化x轴刻度
            if len(pred) > 20:
                ax.set_xticks(np.arange(0, len(pred), len(pred)//10))
        
        fig.tight_layout()
        fig.canvas.draw()  # 先绘制图形到画布
        width, height = fig.get_size_inches() * fig.get_dpi()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(int(height), int(width), 3)
        plt.close(fig)
        runner.visualizer.add_image("Pred_L3", img_array, global_step=runner.iter)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: list[BaseDataElement],
                        outputs: list[BaseDataElement]) -> None:
        """
        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """


class L3_Visualizer(BaseViser):
    def __init__(
        self,
        name,
        resize: Sequence[int] | None = None,
        label_text_scale: float = 0.05,
        label_text_thick: float = 1,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.resize = resize
        self.label_text_scale = label_text_scale
        self.label_text_thick = label_text_thick

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Seg3DDataSample | None = None,
        *args,
        **kwargs,
    ) -> None:
        """显示ZY横断面，取X的中心切面，并可视化1D的L3标记数据。

        Args:
            name: 图像的名称
            image (np.ndarray): 要可视化的图像，NdArray (Z, Y, X, C)。
            data_sample (Seg3DDataSample, optional): 要可视化的数据样本。
                - gt_L3 (data:SeriesData): tensor (Z,)
                - pred_sem_seg_L3 (data:SeriesData): tensor (Z,)
        """
        assert image.ndim == 4, (
            f"The input image must be 4D, but got " f"shape {image.shape}."
        )
        Z, Y, X, C = image.shape
        name += f"_zy_plane"
        
        # 取X的中心切面，显示ZY横断面
        center_x = X // 2
        zy_plane = image[:, :, center_x, :].copy()
        zy_plane = (zy_plane / zy_plane.max() * 255).astype(np.uint8)  # (Z, Y, C)
        
        if self.resize is not None:
            zy_plane = cv2.resize(zy_plane, self.resize, interpolation=cv2.INTER_LINEAR)
        
        l3_info = {'gt_L3': data_sample.gt_L3.data.squeeze(0),
                   'pred_L3': data_sample.pred_sem_seg_L3.data.squeeze(0)}
        
        self.draw_L3(name, zy_plane, l3_info, *args, **kwargs)
    
    def draw_L3(self, name, image, l3_info, *args, **kwargs):
        """绘制1D的L3标记在ZY平面上的可视化结果。
        
        Args:
            name: 图像名称
            image: ZY平面的图像 (Z, Y, C)
            l3_info: 包含gt_L3和pred_L3的字典
        """
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image) # 显示ZY平面图像
        
        if 'gt_L3' in l3_info:
            orig_Z = len(l3_info['gt_L3'])
        elif 'pred_L3' in l3_info:
            orig_Z = len(l3_info['pred_L3'])
        else:
            ax.set_title('没有L3数据可用')
            fig.canvas.draw()
            img_array = self.export_fig_to_ndarray(fig)
            self.add_image(name, img_array, step=kwargs.get('step', 0))
            return
        
        # 计算缩放比例，缩放在外层调用进行
        Z, Y, _ = image.shape
        scale_factor = Z / orig_Z
        
        # 创建一个叠加图层用于GT和Pred
        gt_overlay = np.zeros((Z, Y, 4), dtype=np.uint8)  # RGBA
        pred_overlay = np.zeros((Z, Y, 4), dtype=np.uint8)  # RGBA
        
        # 绘制gt_L3
        gt_l3 = l3_info['gt_L3']
        if isinstance(gt_l3, Tensor):
            gt_l3 = gt_l3.cpu().numpy()
        gt_l3_resize_to_img = zoom(gt_l3, zoom=scale_factor, order=0)
        gt_positions = np.where(gt_l3_resize_to_img == 1)[0]
        if len(gt_positions) > 0:
            for scaled_pos in gt_positions:
                # 半透明绿色
                gt_overlay[scaled_pos, :, 0] = 0    # R
                gt_overlay[scaled_pos, :, 1] = 255  # G
                gt_overlay[scaled_pos, :, 2] = 0    # B
                gt_overlay[scaled_pos, :, 3] = 255  # A
        ax.imshow(gt_overlay, alpha=0.5)

        # 绘制pred_L3
        pred_l3 = l3_info['pred_L3']
        if isinstance(pred_l3, Tensor):
            pred_l3 = pred_l3.cpu().numpy()
        pred_l3_resize_to_img = zoom(pred_l3, zoom=scale_factor, order=0)
        pred_positions = np.where(pred_l3_resize_to_img == 1)[0]
        try:
            if len(pred_positions) > 0:
                for scaled_pos in pred_positions:
                    # 半透明红色
                    pred_overlay[scaled_pos, :, 0] = 255  # R
                    pred_overlay[scaled_pos, :, 1] = 0    # G
                    pred_overlay[scaled_pos, :, 2] = 0    # B
                    pred_overlay[scaled_pos, :, 3] = 255  # 不透明
            ax.imshow(pred_overlay, alpha=0.5)
        except Exception as e:
            pdb.set_trace()
            pass
        
        # 图像元素
        legend_elements = [Patch(facecolor='green', alpha=0.3, label='GT L3')]
        legend_elements.append(Patch(facecolor='red', label='Pred L3'))
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        fig.tight_layout()
        
        # 输出
        fig.canvas.draw()
        img_array = self.export_fig_to_ndarray(fig)
        self.add_image(name, img_array, step=kwargs.get('step', 0))


class L3Metric(BaseMetric):
    """
    1D场景下的Recall、Precision和HD95计算。
    使用pred_sem_seg_L3和gt_sem_seg_L3分别表示预测和真值。
    """
    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        """
        收集每批数据中的预测和真值。
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg_L3']['data'].squeeze()
            gt_label = data_sample['gt_L3']['data'].squeeze()
            pred_label = pred_label if isinstance(pred_label, np.ndarray) else pred_label.cpu().numpy()
            gt_label = gt_label if isinstance(gt_label, np.ndarray) else gt_label.cpu().numpy()
            assert pred_label.shape == gt_label.shape, (
                f"预测和真值的形状不一致：{pred_label.shape} != {gt_label.shape}")
            
            self.results.append((pred_label, gt_label))

    def compute_metrics(self, results) -> dict[str, float]:
        """
        计算1D场景下的Recall、Precision和HD95。
        """
        ious, recalls, precisions, hd95s, productCompliance = [], [], [], [], []

        for (pred, gt) in results:
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            
            iou = tp / max(tp + fp + fn, 1e-6)
            precision = tp / max(tp + fp, 1e-6)
            recall = tp / max(tp + fn, 1e-6)
            hd95 = self.compute_hd95_1d(pred, gt)

            ious.append(iou)
            recalls.append(recall)
            precisions.append(precision)
            hd95s.append(hd95)
            productCompliance.append(iou>=0.85)

        return {
            'iou': float(np.mean(ious)),
            'Recall': float(np.mean(recalls)),
            'Precision': float(np.mean(precisions)),
            'HD95': float(np.mean(hd95s)),
            'ProductComplianceRatio': float(np.mean(productCompliance))
        }

    def compute_hd95_1d(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        在1D数组中计算Hausdorff Distance的95分位数(HD95)。

        1. 找到pred和gt为1的所有索引，构成前景位置。
        2. 对双方前景位置计算最近距离并合并。
        3. 返回这些距离的95分位数。
        """
        pred_indices = np.where(pred == 1)[0]
        gt_indices = np.where(gt == 1)[0]

        # 若任一前景为空，约定返回0
        if len(pred_indices) == 0 and len(gt_indices) == 0:
            return 0.0
        if len(pred_indices) == 0 or len(gt_indices) == 0:
            return float(max(len(pred), len(gt)))

        # 分别计算 pred->gt, gt->pred 的最近距离
        dist_list = []
        for p in pred_indices:
            dist_list.append(np.min(np.abs(gt_indices - p)))
        for g in gt_indices:
            dist_list.append(np.min(np.abs(pred_indices - g)))

        # 返回95分位数距离
        return float(np.percentile(dist_list, 95))


class gen_L3_label(BaseTransform):
    """
    Required Fields:
        - img: [Z, Y, X, C (May Exist)]
    
    Added Fields:
        - gt_L3: [1, Z]
    """
    def transform(self, results:dict):
        if 'L3_anno' in results:
            Z, Y, X = results['img'].shape[:3]
            mask = np.zeros((1, Z), dtype=np.uint8)
            anno_L3_start, anno_L3_mid, anno_L3_end = results['L3_anno']
            # 鉴影标号是倒序的 NOTE 注意下方start和end调转匹配了，不是简单的减Z
            # anno_L3_end, anno_L3_mid, anno_L3_start = Z - anno_L3_start, Z - anno_L3_mid, Z - anno_L3_end
            mask[0, anno_L3_start:anno_L3_end] = 1
            results['gt_L3'] = mask
            results['seg_fields'].append('gt_L3')
        
        elif 'gt_seg_map' in results:
            results['gt_L3'] = np.any(results['gt_seg_map'], axis=(1,2))[None].astype(np.uint8) # pyright:ignore
            results['seg_fields'].append('gt_L3')
        
        return results


class ForegroundSlicesMetric(IoUMetric_PerClass):
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes']) # pyright:ignore
        for data_sample in data_samples:
            pred_label:Tensor = data_sample['pred_sem_seg']['data'].squeeze()         # [Z, Y, X]
            label:Tensor = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label) # [Z, Y, X]
            foreground_Zs = label.any(dim=(1,2)).argwhere()
            pred_label = pred_label[foreground_Zs]
            label = label[foreground_Zs]
            batch_result = self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
            self.results.append(batch_result)



# model's >= 1.8 should use BodyCompositionMetric.
class BodyCompositionMetric(ForegroundSlicesMetric):
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        override dataset_meta's class definition.
        add two merged classes for body composition,
        thus aligning to the original class definition,
        which does not tell inter-muscle fat.
        """
        if CLASS_MAP[1] not in self.dataset_meta["classes"]:
            self.dataset_meta["classes"].insert(-1, CLASS_MAP[1])
        if CLASS_MAP[2] not in self.dataset_meta["classes"]:
            self.dataset_meta["classes"].insert(-1, CLASS_MAP[2])
        return super().process(data_batch, data_samples)

    def intersect_and_union(self, pred_label: Tensor, label: Tensor, num_classes: int, ignore_index: int):
        # 遮盖掉ignore_index
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        
        pred_one_hot = torch.nn.functional.one_hot(pred_label.to(torch.int64), num_classes=num_classes).bool()
        label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=num_classes).bool()

        # 合并新类
        # PsoasMajor-Muscle + PsoasMajor-Fat = Psoas Major Muscle
        pred_one_hot[:, -2] = pred_one_hot[:, 1] | pred_one_hot[:, 5]
        label_one_hot[:, -2] = label_one_hot[:, 1] | label_one_hot[:, 5]
        # OtherSkeletal-Muscle + OtherSkeletal-Fat = Other Skeletal Muscle
        pred_one_hot[:, -1] = pred_one_hot[:, 2] | pred_one_hot[:, 6]
        label_one_hot[:, -1] = label_one_hot[:, 2] | label_one_hot[:, 6]

        # 逐通道计算
        intersect = (pred_one_hot & label_one_hot).sum(dim=0).cpu()
        area_pred_label = pred_one_hot.sum(dim=0).cpu()
        area_label = label_one_hot.sum(dim=0).cpu()
        area_union = area_pred_label + area_label - intersect
        
        return intersect, area_union, area_pred_label, area_label