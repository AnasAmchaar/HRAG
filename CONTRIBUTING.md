# Contributing to H-RAG

Thank you for your interest in contributing to H-RAG! This guide will help you get started.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/AnasAmchaar/HRAG.git
cd H-RAG

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,demo]"

# (Optional) Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests to confirm everything works
pytest

# Run linter
ruff check hrag/

# Run type checker
mypy hrag/
```

## Making Changes

### Branch Naming

Create a descriptive branch for your changes:

```bash
git checkout -b feature/add-docx-support
git checkout -b fix/clustering-edge-case
git checkout -b docs/improve-api-reference
```

### Commit Messages

Write clear, concise commit messages:

```
feat: add DOCX file format support to chunker
fix: handle edge case when all vectors are noise in HDBSCAN
docs: add configuration reference to README
test: add tests for empty document ingestion
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hrag --cov-report=html

# Run a specific test file
pytest tests/test_chunker.py

# Run a specific test
pytest tests/test_chunker.py::TestChunking::test_single_chunk

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Use pytest fixtures from `conftest.py` where possible
- Aim for clear, descriptive test names that explain the expected behavior

```python
class TestMyFeature:
    def test_handles_empty_input(self):
        """Empty input should return an empty result without errors."""
        result = my_function([])
        assert result == []
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Key Rules

- **Line length**: 100 characters
- **Imports**: Sorted with `isort` (via Ruff)
- **Type hints**: Use wherever practical
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Logging**: Use `logging` instead of `print()` in library code

### Running the Linter

```bash
# Check for issues
ruff check hrag/

# Auto-fix issues
ruff check hrag/ --fix

# Format code
ruff format hrag/
```

## Pull Request Process

1. **Update your fork** with the latest `main` branch
2. **Create a feature branch** from `main`
3. **Make your changes** with clear, focused commits
4. **Add or update tests** for your changes
5. **Run the full test suite** and ensure all tests pass
6. **Update documentation** if needed (README, docstrings, etc.)
7. **Open a Pull Request** with a clear description of your changes

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Linter passes (`ruff check hrag/`)
- [ ] New code has docstrings and type hints
- [ ] Changes are documented (README, CHANGELOG if needed)
- [ ] Commit messages are clear and descriptive

## Reporting Issues

### Bug Reports

When filing a bug report, please include:

1. **Python version** (`python --version`)
2. **H-RAG version** (`hrag --version`)
3. **Operating system**
4. **Steps to reproduce** the issue
5. **Expected vs actual behavior**
6. **Error messages or tracebacks**

### Feature Requests

For feature requests, please describe:

1. **The problem** you're trying to solve
2. **Your proposed solution** (if any)
3. **Alternatives** you've considered

## Questions?

If you have questions about contributing, feel free to open a [Discussion](https://github.com/YOUR_USERNAME/H-RAG/discussions) on GitHub.
