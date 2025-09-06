# Contributing to Drone Vehicle Detection System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the drone vehicle detection and counting system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing](#testing)
5. [Submitting Changes](#submitting-changes)
6. [Feature Requests](#feature-requests)
7. [Bug Reports](#bug-reports)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of computer vision and machine learning
- Familiarity with OpenCV and PyTorch/YOLOv8

### First Steps

1. Fork the repository
2. Clone your fork locally
3. Run the setup script: `python setup.py`
4. Run tests to ensure everything works: `python -m pytest tests/`

## Development Setup

### Virtual Environment

Always use a virtual environment for development:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### IDE Setup

Recommended IDE configurations:

#### VS Code
- Install Python extension
- Install Pylint or Flake8 for linting
- Configure format on save with Black

#### PyCharm
- Configure Python interpreter to use virtual environment
- Enable PEP 8 code style inspections
- Set up pytest as test runner

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **String quotes**: Use double quotes for strings
- **Imports**: Group imports according to PEP 8
- **Type hints**: Use type hints for all public functions

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_private_method`
- **Module names**: `lowercase` or `snake_case`

### Code Structure

```python
"""
Module docstring describing the purpose.

Author: Your Name
"""

import standard_library
import third_party_library

from local_package import local_module


class ExampleClass:
    """Class docstring."""
    
    def __init__(self, param: str):
        """Initialize with parameter."""
        self.param = param
    
    def public_method(self, arg: int) -> bool:
        """Public method with type hints and docstring."""
        return self._private_method(arg)
    
    def _private_method(self, arg: int) -> bool:
        """Private method implementation."""
        return arg > 0
```

### Documentation

- Use Google-style docstrings
- Document all public classes, methods, and functions
- Include type information in docstrings
- Provide usage examples for complex functions

Example:
```python
def detect_vehicles(frame: np.ndarray, confidence: float = 0.5) -> List[Detection]:
    """
    Detect vehicles in a video frame.
    
    Args:
        frame: Input image as numpy array with shape (H, W, 3)
        confidence: Minimum confidence threshold for detections
        
    Returns:
        List of Detection objects containing bounding boxes and metadata
        
    Raises:
        ValueError: If frame is not a valid image array
        
    Example:
        >>> frame = cv2.imread('traffic.jpg')
        >>> detections = detect_vehicles(frame, confidence=0.7)
        >>> print(f"Found {len(detections)} vehicles")
    """
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── test_detection/
│   ├── test_vehicle_detector.py
│   └── test_yolo_integration.py
├── test_tracking/
│   └── test_multi_tracker.py
├── test_utils/
│   ├── test_config_manager.py
│   └── test_video_processor.py
└── fixtures/
    └── sample_data.py
```

### Writing Tests

- Use `unittest` or `pytest`
- Test both positive and negative cases
- Mock external dependencies (YOLO models, file I/O)
- Include integration tests for main workflows

Example test:
```python
def test_vehicle_detection_with_mock_model(self):
    """Test vehicle detection with mocked YOLO model."""
    with patch('src.detection.vehicle_detector.YOLO') as mock_yolo:
        # Setup mock
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        # Test
        detector = VehicleDetector(self.config)
        detections = detector.detect(self.test_frame)
        
        # Assertions
        self.assertIsInstance(detections, list)
        mock_model.assert_called_once()
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_vehicle_detector.py

# Run with coverage
python -m pytest tests/ --cov=src/

# Run with verbose output
python -m pytest tests/ -v
```

## Submitting Changes

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make your changes following the style guidelines
3. Add or update tests for your changes
4. Ensure all tests pass: `python -m pytest tests/`
5. Update documentation if needed
6. Commit with clear, descriptive messages
7. Push to your fork and create a pull request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add vehicle speed estimation feature

- Implement speed calculation based on tracking data
- Add configuration options for speed estimation
- Include unit tests for speed calculations
- Update documentation with speed feature usage

Closes #123
```

Format: `type(scope): description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

## Feature Requests

### Before Requesting

1. Check existing issues and pull requests
2. Consider if the feature fits the project scope
3. Think about implementation complexity

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Problem Solved
What problem does this feature solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches you've considered.

## Additional Context
Screenshots, examples, or other relevant information.
```

## Bug Reports

### Before Reporting

1. Check if the bug is already reported
2. Try to reproduce the issue
3. Gather relevant information

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Project version: [e.g., 1.0.0]
- GPU: [e.g., NVIDIA RTX 3080, None]

## Additional Context
Logs, screenshots, or other relevant information.
```

## Development Guidelines

### Performance Considerations

- Profile code for performance bottlenecks
- Use efficient algorithms for video processing
- Consider memory usage for large videos
- Optimize for both CPU and GPU processing

### Error Handling

- Use appropriate exception types
- Provide meaningful error messages
- Log errors with sufficient context
- Handle edge cases gracefully

### Configuration

- Make features configurable when possible
- Validate configuration values
- Provide sensible defaults
- Document configuration options

### Dependencies

- Minimize external dependencies
- Pin dependency versions
- Consider compatibility across platforms
- Document any special installation requirements

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on technical discussions

### Communication

- Use clear, professional language
- Provide context for discussions
- Reference relevant issues/PRs
- Be patient with response times

## Getting Help

- Check the documentation first
- Search existing issues
- Ask questions in discussions
- Provide detailed context when asking for help

Thank you for contributing to the Drone Vehicle Detection System!
