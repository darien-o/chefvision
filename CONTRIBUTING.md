# Contributing to Multi-Model Fusion Detection System

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing to the multi-model fusion detection system.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/chefvision.git
   cd chefvision
   ```

2. **Set Up Python Environment**
   
   Ensure you have Python 3.8 or higher installed:
   ```bash
   python --version
   ```

3. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install Dependencies**
   
   Install production dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

5. **Download Models**
   
   Download the required models:
   ```bash
   python download_models.py
   ```

6. **Verify Installation**
   
   Run the test suite to ensure everything is set up correctly:
   ```bash
   pytest
   ```

## Development Workflow

We follow a standard fork-and-pull-request workflow:

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy of the repository.

### 2. Create a Feature Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features (e.g., `feature/add-yolov9-support`)
- `fix/` - Bug fixes (e.g., `fix/iou-calculation-error`)
- `docs/` - Documentation updates (e.g., `docs/update-installation-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-fusion-algorithm`)
- `test/` - Test additions or modifications (e.g., `test/add-fusion-tests`)

### 3. Make Your Changes

- Write clear, concise code following our coding standards (see below)
- Add tests for new features or bug fixes
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

Before submitting, ensure all tests pass:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_fusion_algorithm.py

# Run linting
flake8 .

# Run type checking
mypy .

# Run code formatting check
black --check .
```

### 5. Commit Your Changes

Follow our commit message conventions (see below).

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Submit a Pull Request

- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your fork and branch
- Fill out the pull request template
- Link any related issues

## Coding Standards

We maintain high code quality standards to ensure maintainability and consistency.

### Code Style

We use the following tools to enforce code style:

#### Black (Code Formatting)

All Python code must be formatted with Black:

```bash
# Format all files
black .

# Check formatting without making changes
black --check .
```

Configuration (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py38']
```

#### Flake8 (Linting)

Code must pass flake8 linting:

```bash
flake8 .
```

Configuration (in `.flake8` or `setup.cfg`):
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv,build,dist
```

#### Mypy (Type Checking)

Use type hints and ensure mypy passes:

```bash
mypy .
```

Configuration (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Code Quality Guidelines

- **Type Hints**: Use type hints for all function signatures
  ```python
  def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
      pass
  ```

- **Docstrings**: Use Google-style docstrings for all public functions and classes
  ```python
  def fusion_detection(detections: list, iou_threshold: float = 0.5) -> list:
      """Fuse multiple model detections using weighted voting.
      
      Args:
          detections: List of detection results from multiple models
          iou_threshold: IoU threshold for spatial clustering (default: 0.5)
          
      Returns:
          List of fused detection results
          
      Raises:
          ValueError: If detections list is empty or iou_threshold is invalid
      """
      pass
  ```

- **Naming Conventions**:
  - Variables and functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

- **Code Organization**:
  - Keep functions focused and under 50 lines when possible
  - Use meaningful variable names
  - Avoid deep nesting (max 3-4 levels)
  - Extract complex logic into separate functions

- **Error Handling**:
  - Use specific exception types
  - Provide descriptive error messages
  - Handle errors at appropriate levels

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build config, etc.)
- `perf`: Performance improvements

### Examples

```
feat(fusion): add support for YOLOv9 models

Implement YOLOv9 model loading and inference integration
with the existing fusion algorithm. Includes configuration
updates and model weight management.

Closes #123
```

```
fix(iou): correct intersection calculation for edge cases

Fix IoU calculation when boxes have zero width or height.
Add validation to prevent division by zero.

Fixes #456
```

```
docs(readme): update installation instructions for Windows

Add Windows-specific setup steps and troubleshooting guide.
```

### Guidelines

- Use imperative mood ("add feature" not "added feature")
- Keep subject line under 72 characters
- Capitalize subject line
- No period at the end of subject line
- Separate subject from body with blank line
- Wrap body at 72 characters
- Use body to explain what and why, not how
- Reference issues and pull requests in footer

## Pull Request Process

### Before Submitting

Ensure your pull request meets these requirements:

- [ ] Code follows our coding standards (Black, flake8, mypy)
- [ ] All tests pass (`pytest`)
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main branch

### Pull Request Template

When you create a pull request, fill out the template completely:

```markdown
## Description
Brief description of changes

## Related Issues
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: Maintainers review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, maintainers will merge

### After Merge

- Delete your feature branch
- Pull the latest changes from main
- Update your fork

## Testing Requirements

All contributions must include appropriate tests.

### Test Types

1. **Unit Tests**: Test individual functions and methods
   ```python
   def test_calculate_iou():
       box1 = (0, 0, 10, 10)
       box2 = (5, 5, 15, 15)
       iou = calculate_iou(box1, box2)
       assert 0 < iou < 1
   ```

2. **Integration Tests**: Test component interactions
   ```python
   def test_fusion_pipeline():
       detections = load_sample_detections()
       result = fusion_detection(detections)
       assert len(result) > 0
   ```

3. **Property-Based Tests**: Test properties across many inputs
   ```python
   from hypothesis import given, strategies as st
   
   @given(st.tuples(st.integers(0, 1000), st.integers(0, 1000),
                    st.integers(0, 1000), st.integers(0, 1000)))
   def test_iou_bounds(box):
       iou = calculate_iou(box, box)
       assert iou == 1.0
   ```

### Test Coverage

- Aim for at least 80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_fusion_algorithm.py

# Run tests matching pattern
pytest -k "test_iou"

# Run with verbose output
pytest -v
```

## Documentation Updates

Documentation is as important as code. When making changes:

### When to Update Documentation

- **New Features**: Add usage examples and API documentation
- **Bug Fixes**: Update if behavior changes
- **Configuration Changes**: Update relevant guides
- **Breaking Changes**: Clearly document in CHANGELOG.md

### Documentation Files to Consider

- `README.md`: Update if user-facing changes
- `docs/USAGE.md`: Add usage examples
- `docs/API.md`: Document new API functions
- `docs/INSTALLATION.md`: Update setup instructions if needed
- `CHANGELOG.md`: Always update for releases
- Inline code comments: Update for complex logic

### Documentation Style

- Use clear, concise language
- Include code examples
- Use proper markdown formatting
- Add diagrams for complex concepts (Mermaid format)
- Keep audience in mind (developers with basic Python/ML knowledge)

### Example Documentation Update

When adding a new feature:

1. Update `README.md` with brief description
2. Add detailed usage example to `docs/USAGE.md`
3. Document API in `docs/API.md`
4. Add entry to `CHANGELOG.md` under "Unreleased"
5. Update inline code comments

## Getting Help

If you need help or have questions:

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `docs/` directory
- **Examples**: Review the `examples/` directory

## Recognition

Contributors will be recognized in:

- GitHub contributors list
- CHANGELOG.md for significant contributions
- README.md acknowledgments section

Thank you for contributing to making this project better!
