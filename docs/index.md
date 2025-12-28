![LOGO](ITKIT-LOGO.png)

# ITKIT Documentation

Welcome to the ITKIT documentation! ITKIT is a user-friendly toolkit built on `SimpleITK` and `Python`, designed for common data preprocessing operations in data-driven CT medical image analysis.

## 📖 Table of Contents

### Getting Started
- **[Installation Guide](installation.md)** - Install ITKIT and its dependencies
- **[Quick Start](quickstart.md)** - Get started with basic usage and examples
- **[Dataset Structure](dataset_structure.md)** - Understand the required dataset format

### Core Features
- **[Preprocessing Tools](preprocessing.md)** - Comprehensive guide to ITK preprocessing commands
  - Image checking and validation
  - Resampling and orientation
  - Patch extraction
  - Data augmentation
  - Label extraction
  - Format conversion

### Advanced Topics
- **[Framework Integration](framework_integration.md)** - Integration with deep learning frameworks
  - OpenMMLab extensions
  - MONAI integration
  - TorchIO integration
  - PyTorch Lightning support
  
- **[Neural Network Models](models.md)** - State-of-the-art segmentation models
  - Transformer-based models (SegFormer, UNETR, DA-TransUNet)
  - State space models (VMamba, SwinUMamba, SegMamba)
  - CNN-based models (MedNeXt, UNet3+, DconnNet)

- **[Supported Datasets](datasets.md)** - Dataset conversion scripts
  - AbdomenCT-1K, BraTS 2024, KiTS23
  - FLARE 2022/2023, TotalSegmentator
  - LiTS, LUNA16, CTSpine1K
  - And more...

### Reference
- **[API Reference](api_reference.md)** - Detailed API documentation
  - IO toolkit (SimpleITK, DICOM, NIfTI)
  - Dataset classes
  - PyTorch Lightning extensions
  - MMEngine extensions
  - Loss functions and utilities

### Community
- **[Contributing Guide](contributing.md)** - How to contribute to ITKIT
  - Development setup
  - Code style guidelines
  - Submission process
  - Release policy

- **[FAQ & Troubleshooting](faq.md)** - Common issues and solutions
  - Installation problems
  - Usage questions
  - Performance tips
  - Framework integration issues

## 🚀 Key Features

- **🔧 Feasible Operations**: Simple command-line interface for complex ITK operations
- **🖥️ GUI Support**: PyQt6-based graphical interface for easier interaction
- **🔌 Framework Integration**: Seamlessly works with MONAI, TorchIO, and OpenMMLab
- **🧠 Comprehensive Models**: State-of-the-art medical segmentation networks
- **📊 Multiple Datasets**: Conversion scripts for 12+ popular medical imaging datasets
- **⚡ High Performance**: Multiprocessing support for faster preprocessing
- **🎨 Flexible**: Works with multiple file formats (MHA, NIfTI, NRRD, DICOM)

## 🎯 Quick Navigation

**New to ITKIT?**
1. Start with [Installation Guide](installation.md)
2. Follow the [Quick Start](quickstart.md) tutorial
3. Learn about [Dataset Structure](dataset_structure.md)

**Ready to preprocess data?**
- See [Preprocessing Tools](preprocessing.md) for command documentation
- Use the GUI: `pip install "itkit[gui]"` then `itkit-app`

**Building models?**
- Explore [Framework Integration](framework_integration.md)
- Check available [Models](models.md)
- Review [API Reference](api_reference.md)

**Need help?**
- Check [FAQ & Troubleshooting](faq.md)
- See [Contributing Guide](contributing.md) to report issues

## 📝 Citation

If you use ITKIT in your research, please cite:

```bibtex
@misc{ITKIT,
    author = {Yiqin Zhang},
    title = {ITKIT: Feasible Medical Image Operation based on SimpleITK API},
    year = {2025},
    url = {https://github.com/MGAMZ/ITKIT}
}
```

## 📧 Contact

For questions or suggestions, reach out at: [312065559@qq.com](mailto:312065559@qq.com)

## 📄 License

ITKIT is released under the [MIT License](https://github.com/MGAMZ/ITKIT/blob/main/LICENSE).
