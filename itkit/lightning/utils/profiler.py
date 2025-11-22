import gc
import json
import os
from datetime import datetime

import psutil
import torch


class MemorySnapshotter:
    """内存快照工具 - 记录RAM和VRAM使用情况"""

    def __init__(self, output_dir: str = "./memory_snapshots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def take_snapshot(self,
                     tag: str = "snapshot",
                     include_tensors: bool = True,
                     include_objects: bool = True) -> dict:
        """
        生成当前内存快照

        Args:
            tag: 快照标签
            include_tensors: 是否包含详细的tensor信息
            include_objects: 是否包含Python对象统计

        Returns:
            包含内存信息的字典
        """
        print(f"📸 [MemorySnapshot] Starting snapshot: {tag}")

        print("  [1/6] Initializing snapshot metadata...")
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
        }

        print("  [2/6] Collecting RAM info...")
        snapshot["ram"] = self._get_ram_info()
        print(f"        ✓ RAM: {snapshot['ram']['process_rss_mb']:.2f} MB")

        print("  [3/6] Collecting VRAM info...")
        snapshot["vram"] = self._get_vram_info()
        if snapshot["vram"].get("available", True):
            total_vram = sum(v.get("allocated_mb", 0) for k, v in snapshot["vram"].items() if k != "available")
            print(f"        ✓ VRAM: {total_vram:.2f} MB allocated")
        else:
            print("        ✓ VRAM: Not available")

        if include_tensors:
            print("  [4/6] Analyzing tensors (this may take a while)...")
            snapshot["tensors"] = self._get_tensor_info()
            print(f"        ✓ Found {snapshot['tensors']['total_count']} tensors, "
                  f"{snapshot['tensors']['total_size_mb']:.2f} MB total")
        else:
            print("  [4/6] Skipping tensor analysis")

        if include_objects:
            print("  [5/6] Analyzing Python objects (this may take a while)...")
            snapshot["objects"] = self._get_object_info()
            print("        ✓ Object analysis complete")
        else:
            print("  [5/6] Skipping object analysis")

        # 保存到文件
        print("  [6/6] Saving snapshot to disk...")
        filename = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"        ✓ JSON saved: {filepath}")

        # 同时生成人类可读的文本报告
        self._save_readable_report(snapshot, filepath.replace('.json', '.txt'))
        print(f"        ✓ Report saved: {filepath.replace('.json', '.txt')}")

        print("📸 [MemorySnapshot] Snapshot complete!\n")
        return snapshot

    def _get_ram_info(self) -> dict:
        """获取RAM使用信息"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()

        return {
            "process_rss_mb": mem_info.rss / 1024 / 1024,  # 实际物理内存
            "process_vms_mb": mem_info.vms / 1024 / 1024,  # 虚拟内存
            "system_total_mb": virtual_mem.total / 1024 / 1024,
            "system_available_mb": virtual_mem.available / 1024 / 1024,
            "system_percent": virtual_mem.percent,
        }

    def _get_vram_info(self) -> dict:
        """获取VRAM使用信息"""
        if not torch.cuda.is_available():
            return {"available": False}

        vram_info = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024 / 1024

            # 获取GPU总内存
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024 / 1024

            vram_info[f"cuda:{i}"] = {
                "device_name": props.name,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated,
                "total_mb": total,
                "utilization_percent": (allocated / total * 100) if total > 0 else 0,
            }

        return vram_info

    def _get_tensor_info(self) -> dict:
        """获取所有Tensor的详细信息"""
        print("        [4.1] Running garbage collection...")
        gc.collect()

        print("        [4.2] Collecting all objects...")
        all_objects = gc.get_objects()
        print(f"        [4.3] Total objects to scan: {len(all_objects)}")

        tensors = []
        print("        [4.4] Scanning for tensors...")
        scanned_count = 0

        for obj in all_objects:
            scanned_count += 1
            if scanned_count % 100000 == 0:
                print(f"              ... scanned {scanned_count}/{len(all_objects)} objects, found {len(tensors)} tensors")

            try:
                if torch.is_tensor(obj):
                    tensor_info = {
                        "dtype": str(obj.dtype),
                        "shape": list(obj.shape),
                        "device": str(obj.device),
                        "size_mb": obj.element_size() * obj.nelement() / 1024 / 1024,
                        "requires_grad": obj.requires_grad,
                    }
                    tensors.append(tensor_info)
            except Exception:
                continue

        print(f"        [4.5] Sorting {len(tensors)} tensors by size...")
        # 按大小排序
        tensors.sort(key=lambda x: x["size_mb"], reverse=True)

        print("        [4.6] Computing statistics...")
        # 统计信息
        total_size = sum(t["size_mb"] for t in tensors)
        cuda_tensors = [t for t in tensors if "cuda" in t["device"]]
        cpu_tensors = [t for t in tensors if "cpu" in t["device"]]

        return {
            "total_count": len(tensors),
            "total_size_mb": total_size,
            "cuda_count": len(cuda_tensors),
            "cuda_size_mb": sum(t["size_mb"] for t in cuda_tensors),
            "cpu_count": len(cpu_tensors),
            "cpu_size_mb": sum(t["size_mb"] for t in cpu_tensors),
            "top_10_largest": tensors[:10],  # 最大的10个tensor
        }

    def _get_object_info(self) -> dict:
        """获取Python对象统计信息"""
        from collections import defaultdict

        print("        [5.1] Running garbage collection...")
        gc.collect()

        print("        [5.2] Collecting all objects...")
        all_objects = gc.get_objects()
        print(f"        [5.3] Total objects to analyze: {len(all_objects)}")

        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)

        print("        [5.4] Analyzing object types and sizes...")
        analyzed_count = 0

        for obj in all_objects:
            analyzed_count += 1
            if analyzed_count % 100000 == 0:
                print(f"              ... analyzed {analyzed_count}/{len(all_objects)} objects")

            try:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
                type_sizes[obj_type] += self._get_object_size(obj)
            except Exception:
                continue

        print("        [5.5] Sorting results...")
        # 按数量排序
        top_by_count = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_by_size = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "top_20_by_count": [{"type": t, "count": c} for t, c in top_by_count],
            "top_20_by_size_mb": [{"type": t, "size_mb": s / 1024 / 1024} for t, s in top_by_size],
        }

    def _get_object_size(self, obj) -> int:
        """估算对象大小 - 使用浅层估算避免性能问题"""
        try:
            import sys

            # 只计算对象本身的大小，不递归计算容器内容
            # 递归计算会导致严重的性能问题
            return sys.getsizeof(obj)
        except Exception:
            return 0

    def _save_readable_report(self, snapshot: dict, filepath: str):
        """保存人类可读的文本报告"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Memory Snapshot Report\n")
            f.write(f"Tag: {snapshot['tag']}\n")
            f.write(f"Timestamp: {snapshot['timestamp']}\n")
            f.write("=" * 80 + "\n\n")

            # RAM信息
            f.write("📊 RAM Usage:\n")
            f.write("-" * 80 + "\n")
            ram = snapshot['ram']
            f.write(f"  Process RSS:      {ram['process_rss_mb']:.2f} MB\n")
            f.write(f"  Process VMS:      {ram['process_vms_mb']:.2f} MB\n")
            f.write(f"  System Total:     {ram['system_total_mb']:.2f} MB\n")
            f.write(f"  System Available: {ram['system_available_mb']:.2f} MB\n")
            f.write(f"  System Usage:     {ram['system_percent']:.1f}%\n\n")

            # VRAM信息
            f.write("🎮 VRAM Usage:\n")
            f.write("-" * 80 + "\n")
            vram = snapshot['vram']
            if vram.get('available', True):
                for device, info in vram.items():
                    if device != 'available':
                        f.write(f"  {device} ({info['device_name']}):\n")
                        f.write(f"    Allocated:     {info['allocated_mb']:.2f} MB\n")
                        f.write(f"    Reserved:      {info['reserved_mb']:.2f} MB\n")
                        f.write(f"    Max Allocated: {info['max_allocated_mb']:.2f} MB\n")
                        f.write(f"    Total:         {info['total_mb']:.2f} MB\n")
                        f.write(f"    Utilization:   {info['utilization_percent']:.1f}%\n\n")
            else:
                f.write("  CUDA not available\n\n")

            # Tensor信息
            if 'tensors' in snapshot:
                f.write("🔢 Tensor Statistics:\n")
                f.write("-" * 80 + "\n")
                tensors = snapshot['tensors']
                f.write(f"  Total Tensors:    {tensors['total_count']}\n")
                f.write(f"  Total Size:       {tensors['total_size_mb']:.2f} MB\n")
                f.write(f"  CUDA Tensors:     {tensors['cuda_count']} ({tensors['cuda_size_mb']:.2f} MB)\n")
                f.write(f"  CPU Tensors:      {tensors['cpu_count']} ({tensors['cpu_size_mb']:.2f} MB)\n\n")

                f.write("  Top 10 Largest Tensors:\n")
                for i, tensor in enumerate(tensors['top_10_largest'], 1):
                    f.write(f"    {i}. {tensor['shape']} {tensor['dtype']} on {tensor['device']}: "
                           f"{tensor['size_mb']:.2f} MB\n")
                f.write("\n")


# 便捷函数
_global_snapshotter = None

def snapshot_memory(tag: str = "snapshot",
                   output_dir: str = "./memory_snapshots",
                   include_tensors: bool = True,
                   include_objects: bool = True) -> dict:
    """
    快速生成内存快照的便捷函数

    使用示例:
        from memory_snapshot import snapshot_memory

        # 在代码任意位置调用
        snapshot_memory("after_model_load")
        snapshot_memory("after_inference")

    注意: include_tensors 和 include_objects 会遍历所有Python对象，
    在大型程序中可能需要几秒到几十秒。如果只需要快速的内存总量统计，
    可以设置为 False。
    """
    global _global_snapshotter

    print(f"\n{'='*80}")
    print(f"🔍 Initializing memory snapshot: '{tag}'")
    print(f"{'='*80}")

    if _global_snapshotter is None:
        print(f"Creating MemorySnapshotter with output_dir: {output_dir}")
        _global_snapshotter = MemorySnapshotter(output_dir)

    return _global_snapshotter.take_snapshot(tag, include_tensors, include_objects)


# 装饰器版本
def memory_snapshot_decorator(tag: str | None = None):
    """
    装饰器版本，自动在函数执行前后记录内存

    使用示例:
        @memory_snapshot_decorator("slide_inference")
        def slide_inference(inputs, model):
            # 你的代码
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_tag = tag or func.__name__

            print(f"📸 Taking memory snapshot before {func_tag}...")
            snapshot_memory(f"{func_tag}_before")

            result = func(*args, **kwargs)

            print(f"📸 Taking memory snapshot after {func_tag}...")
            snapshot_memory(f"{func_tag}_after")

            return result
        return wrapper
    return decorator
