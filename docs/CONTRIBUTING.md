# Contributing to Mathematics for AI

Thank you for your interest in contributing! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/mathematics-for-ai.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Make your changes
2. Run tests: `pytest tests/`
3. Run linting: `flake8 src/ tests/`
4. Run type checking: `mypy src/`
5. Commit with clear messages
6. Push and create a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep functions focused and small (< 50 lines)
- Use meaningful variable names

### Example Code Style

```python
def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)

    Returns:
        Product matrix (m x p)

    Raises:
        ValueError: If matrices have incompatible shapes
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")

    return np.matmul(A, B)
```

## Testing

- Write tests for new features
- Maintain > 90% code coverage
- Use descriptive test names
- Test edge cases and error conditions

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_linear_algebra.py
```

## Documentation

- Update README.md for new features
- Add docstrings to all public APIs
- Create example notebooks for major features
- Update API documentation

## Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update relevant docs
5. **Changelog**: Add entry if applicable

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
- [ ] Changelog updated
```

## Questions?

Open an issue for any questions or discussions.
