#!/usr/bin/env python3
"""
重构后的重采样脚本使用示例
"""

import subprocess
import os

def run_resample_example():
    """运行重采样脚本的示例"""
    
    # 示例1: 基本使用 - 重采样单个文件夹到指定spacing
    print("=== 示例1: 基本重采样 ===")
    print("python mgamdata/process/itk_resample.py /path/to/source_folder /path/to/output_folder --spacing 1.5 1.0 1.0")
    
    # 示例2: 重采样到指定size
    print("\n=== 示例2: 重采样到指定大小 ===")
    print("python mgamdata/process/itk_resample.py /path/to/source_folder /path/to/output_folder --size 128 256 256")
    
    # 示例3: 递归模式处理子目录
    print("\n=== 示例3: 递归模式 ===")
    print("python mgamdata/process/itk_resample.py /path/to/source_folder /path/to/output_folder --spacing 1.5 -1 1.5 -r")
    
    # 示例4: 使用多进程加速
    print("\n=== 示例4: 多进程模式 ===")
    print("python mgamdata/process/itk_resample.py /path/to/source_folder /path/to/output_folder --spacing 1.5 1.0 1.0 --mp --workers 4")
    
    # 示例5: 混合spacing和size (不同维度)
    print("\n=== 示例5: 混合重采样 ===")
    print("python mgamdata/process/itk_resample.py /path/to/source_folder /path/to/output_folder --spacing 1.5 -1 -1 --size -1 256 256")

if __name__ == "__main__":
    print("重构后的重采样脚本使用示例")
    print("=" * 50)
    print("新的命令行接口:")
    print("  - 位置参数: source_folder dest_folder")
    print("  - -r/--recursive: 递归处理子目录")
    print("  - --spacing: 目标spacing (ZYX顺序)")
    print("  - --size: 目标size (ZYX顺序)")
    print("  - --mp: 启用多进程")
    print("  - --workers: 工作进程数")
    print()
    
    run_resample_example()
