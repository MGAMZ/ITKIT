# Contributing Guide

Thank you for your interest in contributing to ITKIT! This guide will help you get started with development and contributions.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/MGAMZ/ITKIT.git
   cd ITKIT
   ```

2. **Install development dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs ITKIT in editable mode with all development tools including:

   - pytest (testing)
   - black, isort, autopep8 (code formatting)
   - mypy, pyright (type checking)
   - pylint (linting)
   - pre-commit (git hooks)

3. **Set up pre-commit hooks:**

   ```bash
   pre-commit install
   ```

   This ensures code is automatically formatted and checked before commits.

## What to Contribute

### Areas for Contribution

1. **Bug Fixes:** Fix reported bugs or issues you encounter

2. **New Features:**
   - New preprocessing operations
   - Additional neural network models
   - Dataset conversion scripts
   - Framework integrations

3. **Documentation:**
   - Improve existing docs
   - Add examples
   - Fix typos
   - Translate documentation

4. **Tests:**
   - Increase test coverage
   - Add edge case tests
   - Add integration tests

5. **Performance:**
   - Optimize slow operations
   - Improve memory usage
   - Add parallelization

### Adding New Models

To add a new neural network model:

1. Create model file in `itkit/models/`
2. Implement model class with PyTorch
3. Add documentation in `docs/models.md`
4. Include reference paper and citation
5. Add usage examples
6. Create unit tests

Example structure:

```python
# itkit/models/my_new_model.py

import torch
import torch.nn as nn

class MyNewModel(nn.Module):
    """
    Brief description of the model.

    Reference: Author et al., "Paper Title", Conference/Journal Year.
    """

    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        # Model architecture

    def forward(self, x):
        # Forward pass
        return output
```

### Adding Dataset Conversion Scripts

To add a new dataset conversion script:

1. Create folder: `itkit/dataset/<dataset_name>/`
2. Add conversion script: `convert_<format>.py`
3. Document in `docs/datasets.md`
4. Include dataset reference and citation
5. Add example usage

Script should:

- Convert to ITKIT format (image/ and label/ folders)
- Preserve metadata
- Handle edge cases
- Provide progress feedback

### Adding Preprocessing Tools

To add a new preprocessing command:

1. Create script in `itkit/process/`
2. Add entry point in `pyproject.toml`
3. Create documentation in `docs/itk_*.md` and update `mkdocs.yml`
4. Add tests in `tests/`
5. Support common flags (`--mp`, `--help`)

## Getting Help

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Email:** Contact maintainer at [312065559@qq.com](mailto:312065559@qq.com)
- **Discussions:** Use GitHub Discussions for questions
