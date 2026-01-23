"""
Example script demonstrating ITKIT 3D Slicer extension usage.

This script shows how to:
1. Load a volume in Slicer
2. Initialize the ITKIT Inference module
3. Run inference programmatically
4. Save the results

Usage:
- Load this script in 3D Slicer's Python console
- Or run it as: Slicer --python-script example_usage.py
"""

import slicer
import os


def example_mmengine_inference():
    """Example: Run inference using MMEngine backend."""
    
    print("=" * 60)
    print("ITKIT Slicer Extension - MMEngine Backend Example")
    print("=" * 60)
    
    # 1. Load a test volume (you can replace this with your own data)
    # For this example, we'll load sample data from Slicer
    print("\n[1/5] Loading test volume...")
    try:
        import SampleData
        volumeNode = SampleData.downloadSample('CTChest')
        print(f"✓ Loaded volume: {volumeNode.GetName()}")
    except Exception as e:
        print(f"✗ Failed to load sample data: {e}")
        print("Please load a volume manually before running this script.")
        return
    
    # 2. Get the ITKIT Inference module
    print("\n[2/5] Initializing ITKIT Inference module...")
    try:
        # Get module logic
        moduleWidget = slicer.modules.itkitinference.widgetRepresentation().self()
        logic = moduleWidget.logic
        print("✓ Module initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize module: {e}")
        print("Make sure the ITKIT Inference extension is installed and loaded.")
        return
    
    # 3. Configure inference parameters
    print("\n[3/5] Configuring inference parameters...")
    
    # IMPORTANT: Replace these paths with your actual model files
    configPath = "/path/to/your/config.py"
    checkpointPath = "/path/to/your/checkpoint.pth"
    
    if not os.path.exists(configPath):
        print(f"✗ Config file not found: {configPath}")
        print("Please update the configPath variable with your actual config file.")
        return
    
    if not os.path.exists(checkpointPath):
        print(f"✗ Checkpoint file not found: {checkpointPath}")
        print("Please update the checkpointPath variable with your actual checkpoint.")
        return
    
    # Inference configuration
    patchSize = (96, 96, 96)  # Adjust based on your model and GPU memory
    patchStride = (48, 48, 48)  # Half of patch size for good overlap
    fp16 = True  # Use half precision for faster inference
    forceCpu = False  # Set to True if GPU memory is limited
    
    print(f"  - Config: {configPath}")
    print(f"  - Checkpoint: {checkpointPath}")
    print(f"  - Patch size: {patchSize}")
    print(f"  - Patch stride: {patchStride}")
    print(f"  - FP16: {fp16}")
    
    # 4. Run inference
    print("\n[4/5] Running inference...")
    print("(This may take several minutes depending on volume size and GPU)")
    
    try:
        segmentationNode = logic.process(
            inputVolume=volumeNode,
            outputSegmentation=None,  # Create new segmentation node
            backend="MMEngine",
            configPath=configPath,
            checkpointPath=checkpointPath,
            patchSize=patchSize,
            patchStride=patchStride,
            fp16=fp16,
            forceCpu=forceCpu,
            progressCallback=lambda p, m: print(f"  Progress: {p}% - {m}")
        )
        print(f"✓ Inference completed successfully!")
        print(f"✓ Output segmentation: {segmentationNode.GetName()}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Display results
    print("\n[5/5] Displaying results...")
    
    # Show segmentation in slice views
    segmentationNode.CreateDefaultDisplayNodes()
    
    # Adjust view to show the segmentation
    slicer.util.setSliceViewerLayers(
        background=volumeNode,
        foreground=None,
        label=None,
        foregroundOpacity=0.5
    )
    
    # Center views on the volume
    slicer.util.resetSliceViews()
    
    print("✓ Results displayed in slice views")
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use Segment Editor to refine the segmentation")
    print("2. Save results: File → Save")
    print("3. Visualize in 3D view")


def example_onnx_inference():
    """Example: Run inference using ONNX backend."""
    
    print("=" * 60)
    print("ITKIT Slicer Extension - ONNX Backend Example")
    print("=" * 60)
    
    # 1. Load a test volume
    print("\n[1/5] Loading test volume...")
    try:
        import SampleData
        volumeNode = SampleData.downloadSample('CTChest')
        print(f"✓ Loaded volume: {volumeNode.GetName()}")
    except Exception as e:
        print(f"✗ Failed to load sample data: {e}")
        return
    
    # 2. Get the ITKIT Inference module
    print("\n[2/5] Initializing ITKIT Inference module...")
    try:
        moduleWidget = slicer.modules.itkitinference.widgetRepresentation().self()
        logic = moduleWidget.logic
        print("✓ Module initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize module: {e}")
        return
    
    # 3. Configure inference parameters
    print("\n[3/5] Configuring inference parameters...")
    
    # IMPORTANT: Replace this path with your actual ONNX model file
    onnxPath = "/path/to/your/model.onnx"
    
    if not os.path.exists(onnxPath):
        print(f"✗ ONNX model not found: {onnxPath}")
        print("Please update the onnxPath variable with your actual ONNX model.")
        return
    
    # Note: For ONNX models, patch size is typically fixed in the model
    # You can still configure stride and other parameters
    patchStride = None  # Use model defaults
    fp16 = False  # ONNX runtime handles precision
    forceCpu = False
    
    print(f"  - Model: {onnxPath}")
    print(f"  - Patch stride: {patchStride or 'auto'}")
    
    # 4. Run inference
    print("\n[4/5] Running inference...")
    
    try:
        segmentationNode = logic.process(
            inputVolume=volumeNode,
            outputSegmentation=None,
            backend="ONNX",
            configPath=None,  # Not needed for ONNX
            checkpointPath=onnxPath,
            patchSize=None,  # Use model defaults
            patchStride=patchStride,
            fp16=fp16,
            forceCpu=forceCpu,
            progressCallback=lambda p, m: print(f"  Progress: {p}% - {m}")
        )
        print(f"✓ Inference completed successfully!")
        print(f"✓ Output segmentation: {segmentationNode.GetName()}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Display results
    print("\n[5/5] Displaying results...")
    segmentationNode.CreateDefaultDisplayNodes()
    slicer.util.setSliceViewerLayers(background=volumeNode)
    slicer.util.resetSliceViews()
    
    print("✓ Results displayed")
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def batch_processing_example():
    """Example: Process multiple volumes in batch."""
    
    print("=" * 60)
    print("ITKIT Slicer Extension - Batch Processing Example")
    print("=" * 60)
    
    # Configuration
    inputDir = "/path/to/input/volumes"
    outputDir = "/path/to/output/segmentations"
    onnxPath = "/path/to/model.onnx"
    
    # Validate paths
    if not os.path.exists(inputDir):
        print(f"✗ Input directory not found: {inputDir}")
        return
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        print(f"✓ Created output directory: {outputDir}")
    
    if not os.path.exists(onnxPath):
        print(f"✗ Model not found: {onnxPath}")
        return
    
    # Get logic
    try:
        moduleWidget = slicer.modules.itkitinference.widgetRepresentation().self()
        logic = moduleWidget.logic
    except Exception as e:
        print(f"✗ Failed to initialize module: {e}")
        return
    
    # Get list of volumes to process
    volumeFiles = [f for f in os.listdir(inputDir) if f.endswith(('.nii', '.nii.gz', '.mha', '.nrrd'))]
    
    print(f"\nFound {len(volumeFiles)} volumes to process")
    print(f"Input: {inputDir}")
    print(f"Output: {outputDir}")
    print(f"Model: {onnxPath}")
    print("")
    
    # Process each volume
    for i, filename in enumerate(volumeFiles, 1):
        print(f"\n[{i}/{len(volumeFiles)}] Processing: {filename}")
        
        # Load volume
        inputPath = os.path.join(inputDir, filename)
        try:
            volumeNode = slicer.util.loadVolume(inputPath)
            print(f"  ✓ Loaded volume")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
        
        # Run inference
        try:
            segmentationNode = logic.process(
                inputVolume=volumeNode,
                outputSegmentation=None,
                backend="ONNX",
                configPath=None,
                checkpointPath=onnxPath,
                patchSize=None,
                patchStride=None,
                fp16=False,
                forceCpu=False,
                progressCallback=None  # Disable progress for batch
            )
            print(f"  ✓ Inference completed")
        except Exception as e:
            print(f"  ✗ Inference failed: {e}")
            slicer.mrmlScene.RemoveNode(volumeNode)
            continue
        
        # Save segmentation
        outputPath = os.path.join(outputDir, filename.replace('.nii.gz', '_seg.nii.gz'))
        try:
            slicer.util.saveNode(segmentationNode, outputPath)
            print(f"  ✓ Saved to: {outputPath}")
        except Exception as e:
            print(f"  ✗ Failed to save: {e}")
        
        # Clean up
        slicer.mrmlScene.RemoveNode(volumeNode)
        slicer.mrmlScene.RemoveNode(segmentationNode)
    
    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    """
    Uncomment one of the examples below to run it.
    """
    
    # Run MMEngine inference example
    # example_mmengine_inference()
    
    # Run ONNX inference example
    # example_onnx_inference()
    
    # Run batch processing example
    # batch_processing_example()
    
    print("Please uncomment one of the example functions to run it.")
    print("Make sure to update the file paths with your actual data.")
