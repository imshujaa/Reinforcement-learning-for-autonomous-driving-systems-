# Contributing to CaRLeT

Thank you for your interest in contributing to CaRLeT! This document provides guidelines and standards for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include a clear description and reproduction steps
- Provide system information (OS, Python version, dependencies)

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain the use case and benefits

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes following our coding standards
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Causal-RL-for-autonomous-driving-systems-.git
cd CaRLeT_replication_package

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -r requirements/dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Style Guide

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 100 characters
- Use type hints for function signatures

### Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include examples in docstrings where appropriate

### Testing

- Write tests for all new code
- Aim for >80% code coverage
- Use pytest for testing
- Run tests before submitting PR:

```bash
pytest tests/ -v --cov=carlet
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense
- Reference issues when applicable

Example:
```
Add support for SAC algorithm

Implements Soft Actor-Critic algorithm for continuous action spaces.
Closes #42
```

## Project Structure

- `carlet/`: Main package code
- `tests/`: Test suite
- `docs/`: Documentation
- `scripts/`: Utility scripts
- `configs/`: Configuration files

## Questions?

Feel free to open an issue or reach out to the maintainers.
