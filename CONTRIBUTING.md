# Contributing to Nirvana

Thank you for your interest in contributing to Nirvana!

## Documentation

### Building Documentation Locally

1. Install dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Serve documentation locally:
   ```bash
   mkdocs serve
   ```

3. Build static site:
   ```bash
   mkdocs build
   ```

The documentation uses Material for MkDocs, similar to DSPy's documentation style.

### Documentation Structure

- `docs/index.md` - Homepage
- `docs/get_started.md` - Quick start guide
- `docs/tutorial.md` - Comprehensive tutorial
- `docs/api_reference.md` - API documentation
- `docs/development.md` - Developer guide

### Documentation Guidelines

- Use clear, concise language
- Include code examples where helpful
- Keep examples up-to-date with the codebase
- Test all code examples before committing
- Use proper markdown formatting

## Code Contributions

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/JunHao-Zhu/nirvana.git
   cd nirvana
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Run `ruff` for linting before committing

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Update documentation if needed
5. Submit a pull request

Thank you for contributing!

