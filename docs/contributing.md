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

## Development Workflow

### Code Style

ITKIT follows these code style guidelines:

- **Python version:** >= 3.10
- **Formatter:** Black with 150 character line length
- **Import sorting:** isort with black profile
- **Type hints:** Encouraged for public APIs
- **Docstrings:** Follow NumPy/Google style

### Running Tests

ITKIT uses pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_io.py

# Run tests with specific marker
pytest -m torch  # Only PyTorch tests
pytest -m itk_process  # Only ITK processing tests

# Run with coverage
pytest --cov=itkit
```

Available test markers:

- `gui`: GUI tests (requires Qt)
- `itk_process`: ITK preprocessing script tests
- `mm`: MMEngine runner tests
- `torch`: PyTorch-dependent tests

### Code Formatting

Format your code before committing:

```bash
# Format with black
black itkit/

# Sort imports
isort itkit/

# Or use pre-commit to run all checks
pre-commit run --all-files
```

### Type Checking

Run type checkers:

```bash
# Pyright
pyright

# Mypy
mypy itkit/
```

## Contribution Guidelines

### Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** if available
3. **Provide clear description** with:
   - Expected behavior
   - Actual behavior
   - Steps to reproduce
   - Environment details (OS, Python version, ITKIT version)
   - Sample code or data if applicable

### Submitting Pull Requests

1. **Create a new branch** for your feature/fix:

```bash
git checkout -b feature/my-new-feature
```

2. **Make your changes** following code style guidelines

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Run tests** to ensure nothing breaks:

```bash
pytest
```

6. **Commit your changes** with clear messages:

```bash
git commit -m "Add feature: description of feature"
```

7. **Push to your fork:**

```bash
git push origin feature/my-new-feature
```

8. **Create a Pull Request** on GitHub

### Pull Request Checklist

- [ ] Code follows ITKIT style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] No unnecessary files included

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
3. Document in `docs/preprocessing.md`
4. Add tests in `tests/`
5. Support common flags (`--mp`, `--help`)

## Release Process

ITKIT follows a specific release policy:

### Release Branches

- Stable releases managed on branches: `v1`, `v2`, `v3`, etc.
- Development occurs on feature branches
- Only merges to release branches trigger releases

### Release Triggers

- Any PR merged into a release branch automatically triggers a new release
- Releases are typically minor version updates (v3.1, v3.2, etc.)
- Tagged accordingly

### Version Numbering

ITKIT uses semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features (backwards compatible)
- **PATCH:** Bug fixes (backwards compatible)

## Communication

### Getting Help

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Email:** Contact maintainer at [312065559@qq.com](mailto:312065559@qq.com)
- **Discussions:** Use GitHub Discussions for questions

### Community Guidelines

- Be respectful and constructive
- Help others when you can
- Follow the code of conduct
- Give credit where due
- Focus on what's best for the project

## License

By contributing to ITKIT, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:

- GitHub contributors list
- Release notes
- CITATION file (for significant contributions)

## Questions?

If you have any questions about contributing, feel free to:

- Open an issue with the "question" label
- Contact the maintainer via email
- Start a discussion on GitHub

Thank you for making ITKIT better!
