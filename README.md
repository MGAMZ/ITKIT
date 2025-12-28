![LOGO](./docs/ITKIT-LOGO.png)

# ITKIT: Feasible Medical Image Operation based on SimpleITK API

[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/) [![SimpleITK >= 2.5.0](https://img.shields.io/badge/SimpleITK-%3E%3D2.5-yellow)](https://github.com/SimpleITK/SimpleITK) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) ![CI Status](https://github.com/MGAMZ/ITKIT/actions/workflows/test.yml/badge.svg)

ITKIT is a comprehensive toolkit for medical image preprocessing and analysis, providing command-line tools, a GUI application, and deep learning framework integrations for CT and MRI image processing.

## ✨ Core Features

- **🔧 Preprocessing Tools**: Check, resample, orient, patch, augment, and convert medical images
- **🖥️ GUI Application**: User-friendly PyQt6 interface for all operations
- **🧠 Neural Networks**: 16+ state-of-the-art segmentation models (SegFormer, MedNeXt, VMamba, etc.)
- **🔌 Framework Support**: Integration with OpenMMLab, MONAI, TorchIO, and PyTorch Lightning
- **📊 Dataset Conversion**: Scripts for 12+ popular medical imaging datasets
- **⚡ High Performance**: Multiprocessing support for faster batch processing

## 🚀 Quick Start

### Installation

```bash
pip install itkit
# Optional: Install GUI support
pip install "itkit[gui]"
```

### Basic Usage

```bash
# Check dataset integrity
itk_check check /path/to/dataset --min-spacing 0.5 0.5 0.5

# Resample images to uniform spacing
itk_resample dataset /path/to/source /path/to/output --spacing 1.0 1.0 1.0 --mp

# Launch GUI application
itkit-app
```

## 📚 Documentation

**Full documentation is available at [docs/](docs/index.md)**

### Quick Links

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Quick Start Tutorial](docs/quickstart.md)** - Get started in 5 minutes
- **[Dataset Structure](docs/dataset_structure.md)** - Required dataset format
- **[Preprocessing Tools](docs/preprocessing.md)** - Complete command reference
- **[Framework Integration](docs/framework_integration.md)** - OpenMMLab, MONAI, TorchIO
- **[Neural Network Models](docs/models.md)** - Available segmentation models
- **[Supported Datasets](docs/datasets.md)** - Dataset conversion scripts
- **[FAQ & Troubleshooting](docs/faq.md)** - Common issues and solutions
- **[Contributing Guide](docs/contributing.md)** - How to contribute

## 🛠️ Command-Line Tools

ITKIT provides several preprocessing commands:

| Command | Description |
|---------|-------------|
| `itk_check` | Validate dataset integrity (spacing, size, pairing) |
| `itk_resample` | Resample images to target spacing or size |
| `itk_orient` | Orient images to standard directions (LPI, RAS, etc.) |
| `itk_patch` | Extract patches for training |
| `itk_aug` | Data augmentation with random rotations |
| `itk_extract` | Extract specific classes from segmentation maps |
| `itk_convert` | Convert between formats (MHA, NIfTI, NRRD) and frameworks (MONAI, TorchIO) |
| `itkit-app` | Launch graphical user interface |
| `mmrun` | Run OpenMMLab experiments |

Use `--help` with any command for detailed usage information.

## 🖼️ GUI Application

![ITKIT GUI](./docs/itkit-gui.png)

Install GUI support and launch:

```bash
pip install "itkit[gui]"
itkit-app

# Adjust DPI if needed
QT_SCALE_FACTOR=2 itkit-app
```

## 📦 Optional Features

ITKIT provides optional dependency groups:

```bash
pip install "itkit[gui]"        # GUI application (PyQt6)
pip install "itkit[advanced]"   # Deep learning frameworks (OpenMMLab)
pip install "itkit[dev]"        # Development tools (pytest, black, mypy)
pip install "itkit[pathology]"  # Pathology image processing
pip install "itkit[onnx]"       # Model deployment (ONNX, TensorRT)
```

## 📖 Citation

If you use ITKIT in your research, please cite:

```bibtex
@misc{ITKIT,
    author = {Yiqin Zhang},
    title = {ITKIT: Feasible Medical Image Operation based on SimpleITK API},
    year = {2025},
    url = {https://github.com/MGAMZ/ITKIT}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## 📄 License

ITKIT is released under the [MIT License](LICENSE).

## 📧 Contact

For questions or suggestions, reach out at: [312065559@qq.com](mailto:312065559@qq.com)

## 🌟 Acknowledgments

ITKIT builds upon:

- [SimpleITK](https://github.com/SimpleITK/SimpleITK) - Medical image processing
- [OpenMMLab](https://github.com/open-mmlab) - Deep learning framework
- [MONAI](https://monai.io/) - Medical imaging AI
- [TorchIO](https://torchio.readthedocs.io/) - Medical image preprocessing

---

**⭐ Star us on GitHub if you find ITKIT useful!**
