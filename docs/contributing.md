# Contributing

Thank you for your interest in contributing to `chronocratic-models`. This guide covers the development workflow and tooling used in the project.

## Development Setup

The project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/chronocratic/chronocratic-models.git
cd chronocratic-models

# Sync the development environment
uv sync
```

## Running Tests

```bash
# Run the full test suite
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml
```

## Linting and Formatting

The project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for linting issues
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Type Checking

The project uses [ty](https://github.com/astral-sh/ty) for static type checking.

```bash
# Run type checking
uv run ty check src/
```

## Building Documentation

```bash
# Install documentation dependencies
uv sync --extra docs

# Build the documentation
uv run sphinx-build -b html docs/ docs/_build/
```

## Adding Changelog Fragments

The project uses [towncrier](https://towncrier.readthedocs.io/) for managing changelog entries. Each PR should include a changelog fragment in the `changelog.d/` directory.

```bash
# Create a fragment (e.g., for a new feature in PR #42)
echo "Added new TimeVAE model for generative time-series encoding." > changelog.d/42.added.md
```

Fragment types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.

```bash
# Verify fragments before merging
uv run towncrier check --compare-with origin/dev
```

See [`changelog.d/README.md`](../changelog.d/README.md) for detailed fragment instructions.

## Code Style

- Use **snake_case** for functions and variables, **PascalCase** for classes.
- Write **Google-style docstrings** for all public functions and classes.
- Use **type hints** for all function signatures and return types.
- Prefer **functional programming patterns** and modular code organization.
- Use **keyword arguments** for all function calls.
