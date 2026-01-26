import glob
import multiprocessing as mp
import os
import traceback
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import zarr
from torch import Tensor
from tqdm import tqdm

VALID_INPUT_EXTS = ['*.mha', '*.nii', '*.nii.gz']


def gen_tasks(args):
    os.makedirs(args.output, exist_ok=True)

    # Collect all available files
    input_files: list[str] = []
    for ext in VALID_INPUT_EXTS:
        input_files.extend(glob.glob(os.path.join(args.input_folder, ext)))
    if not input_files:
        raise ValueError(f"No input files found in {args.input_folder}")

    # Pre-filter out existing output files
    pending = []
    for f in input_files:
        output_path = Path(args.output) / Path(f).name
        if output_path.exists():
            print(f"Skipping existing {output_path}")
        else:
            pending.append(f)
    if not pending:
        print("No new files to process")
        return [[] for _ in range(args.num_proc)]

    # Distribute tasks evenly to each process
    if args.num_proc == 1 or len(pending) == 1:
        return [pending]
    avg = len(pending) // args.num_proc
    rem = len(pending) % args.num_proc
    tasks = []
    idx = 0
    for i in range(args.num_proc):
        cnt = avg + (1 if i < rem else 0)
        tasks.append(pending[idx:idx+cnt])
        idx += cnt
    return tasks


def set_window(image_array:np.ndarray, wl:int, ww:int) -> np.ndarray:
    left = wl - ww/2
    right = wl + ww/2
    image_array = np.clip(image_array.astype(np.int16), left, right)
    image_array = (image_array - left) / ww
    return image_array


def calc_classwise_pred_confidence(seg_logits: Tensor) -> Tensor:
    """
    Calculate prediction confidence (inverse entropy) across all spatial dimensions.

    Args:
        seg_logits (Tensor): Segmentation logits tensor with shape (N, C, Z, Y, X).

    Returns:
        confidence (Tensor): Tensor with shape (N,) representing the mean confidence across all classes.
    """
    assert (C:=seg_logits.size(1)) >= 2, f"Number of classes must be at least 2, got {C}."

    # Compute softmax probabilities
    probs = torch.softmax(seg_logits, dim=1)  # (N, C, Z, Y, X)

    # Calculate entropy: -sum(p * log(p))
    entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=1)  # (N, Z, Y, X)

    # Normalize entropy to [0,1] by dividing by maximum entropy = log(C)
    max_entropy = np.log(C)

    normalized_entropy = entropy / max_entropy
    confidence = 1.0 - normalized_entropy  # (N, Z, Y, X)

    return confidence.mean(dim=(1, 2, 3))  # (N,)


def process_gpu_task(process_id, file_list, args, pred_conf_shared_dict=None):
    # NOTE Local environment setup for each GPU process.
    gpu_id = process_id % args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from itkit.mm.inference import Inferencer_Seg3D, MMEngineInferBackend, ONNXInferBackend
    from itkit.mm.sliding_window import InferenceConfig
    tqdm.write(f"Process {process_id} using GPU {gpu_id}, processing {len(file_list)} files")

    infer_cfg_override = None
    if args.patch_size is not None or args.patch_stride is not None:
        infer_cfg_override = InferenceConfig(
            patch_size=tuple(args.patch_size) if args.patch_size is not None else None,
            patch_stride=tuple(args.patch_stride) if args.patch_stride is not None else None
        )

    if args.backend == "mmengine":
        backend = MMEngineInferBackend(
            cfg_path=args.cfg_path,
            ckpt_path=args.ckpt_path,
            inference_config=infer_cfg_override,
            allow_tqdm=False
        )
        wl = args.wl if args.wl is not None else backend.cfg.get('wl')
        ww = args.ww if args.ww is not None else backend.cfg.get('ww')

    elif args.backend == "onnx":  # onnx
        backend = ONNXInferBackend(
            onnx_path=args.onnx,
            inference_config=infer_cfg_override,
            allow_tqdm=False
        )
        wl, ww = args.wl, args.ww
        if wl is None or ww is None:
            meta = backend.session.get_modelmeta().custom_metadata_map or {}
            if wl is None and 'window_level' in meta:
                wl = int(meta['window_level'])
            if ww is None and 'window_width' in meta:
                ww = int(meta['window_width'])
        if wl is None or ww is None:
            raise ValueError("--backend onnx requires --wl/--ww, or the ONNX must contain metadata: window_level/window_width")

    else:
        raise NotImplementedError(f"Backend {args.backend} not supported.")

    inferencer = Inferencer_Seg3D(
        backend=backend,
        fp16=args.fp16,
        allow_tqdm=False
    )

    pred_confidences = {}
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    for file_path in tqdm(file_list,
                          dynamic_ncols=True,
                          leave=False,
                          mininterval=1,
                          position=process_id,
                          desc=f"Proc{process_id}-GPU{gpu_id}"):
        try:
            # Prepare
            file = Path(file_path)
            itk_image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(itk_image)
            image_array = set_window(image_array, wl, ww)
            image_array = image_array.astype(np.float16 if args.fp16 else np.float32)

            # Inference
            pred_seg_logits, pred_sem_seg = inferencer.Inference_FromNDArray(image_array)
            assert pred_seg_logits.size(0) == 1, "Batch size > 1 not supported in this script."
            assert pred_sem_seg.size(0) == 1, "Batch size > 1 not supported in this script."

            # Save semantic segmentation map
            itk_pred = sitk.GetImageFromArray(pred_sem_seg[0].cpu().numpy())
            itk_pred.CopyInformation(itk_image)
            itk_pred = sitk.DICOMOrient(itk_pred, 'LPI')
            sitk.WriteImage(itk_pred, output_folder/file.name, True)

            # Register Confidence Map
            if pred_conf_shared_dict is not None:
                confidence = calc_classwise_pred_confidence(pred_seg_logits)  # (N,)
                pred_confidences[file.stem] = confidence[0].cpu().item()

            # Save segmentation logits as .npz
            logits_np = pred_seg_logits[0].cpu().numpy().astype(np.float16)
            if args.save_logits:
                zarr.save_array(
                    output_folder / (file.stem+'.zarr'),
                    logits_np,  # pyright: ignore[reportArgumentType]
                    codecs=[{"name": "bytes"},
                            {"name": "blosc",
                             "configuration": {"cname": "lz4",
                                               "clevel": 6}}]
                )

        except Exception as e:
            traceback.print_exc()
            tqdm.write(f"Error processing {file_path}: {e}")

    # Update shared dict with pred_confidences
    if pred_conf_shared_dict is not None:
        pred_conf_shared_dict.update(pred_confidences)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inferencer')
    parser.add_argument('-i', '--input-folder', type=str, required=True, help='Input folder path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder path')

    # Backend options
    parser.add_argument('--backend', type=str, choices=['mmengine', 'onnx'], default='mmengine', help='Inference backend')
    # MMEngine related
    parser.add_argument('-cfg', '--cfg-path', type=str, help='Config file path (required for mmengine)')
    parser.add_argument('-ckpt', '--ckpt-path', type=str, help='Checkpoint file path (required for mmengine)')
    # ONNX related
    parser.add_argument('--onnx', type=str, help='ONNX model path (required for onnx)')

    # Windowing parameters (optional, defaults to config if mmengine)
    parser.add_argument('--wl', type=int, help='Window level')
    parser.add_argument('--ww', type=int, help='Window width')
    # Inference config overrides (optional)
    parser.add_argument('--patch-size', type=int, nargs=3, metavar=('Z', 'Y', 'X'),
                        help='Override inference patch size (Z Y X)')
    parser.add_argument('--patch-stride', type=int, nargs=3, metavar=('Z', 'Y', 'X'),
                        help='Override inference patch stride (Z Y X)')

    # Other options
    parser.add_argument('--num-proc', type=int, default=1, help='Number of processes to use')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use FP16 precision')
    parser.add_argument('--save-logits', action='store_true', default=False)
    parser.add_argument('--save-conf', action='store_true', default=False)

    args = parser.parse_args()

    if args.backend == 'mmengine':
        if not args.cfg_path or not args.ckpt_path:
            parser.error("--backend mmengine requires --cfg-path and --ckpt-path")
    elif args.backend == 'onnx':
        if not args.onnx:
            parser.error("--backend onnx requires --onnx")

    return args


def main():
    args = parse_args()

    # Allocate task
    task_per_process = gen_tasks(args)
    total_tasks = sum(len(t) for t in task_per_process)
    print(f"Found {total_tasks} files to process")

    # Create shared dict for collecting pred_confidences
    if args.save_conf:
        manager = Manager()
        pred_conf_shared_dict = manager.dict()
    else:
        pred_conf_shared_dict = None

    processes = []
    for process_id, file_list in enumerate(task_per_process):
        if not file_list:
            print(f"Process {process_id} has no tasks to process")
            continue
        p = mp.get_context('spawn').Process(
            target = process_gpu_task,
            args = (process_id, file_list, args, pred_conf_shared_dict),
            daemon = True
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect all pred_confidences from pred_conf_shared_dict
    if pred_conf_shared_dict is not None:
        pd.DataFrame.from_dict(
            dict(pred_conf_shared_dict),
            orient='index',
            columns=['Confidence']
        ).to_excel(os.path.join(args.output, 'confidences.xlsx'), sheet_name='Confidences')


if __name__ == '__main__':
    main()
