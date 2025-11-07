# Field Extraction Project

A Python project for processing input documents and ground truth documents, building datasets, training prompting strategies, and returning final datasets and configurations.

## Project Structure

```
field_extraction/
├── data/              # Data directory
│   ├── raw/          # Raw input documents
│   ├── interim/      # Intermediate processed data
│   └── processed/    # Final processed datasets
├── scripts/          # Source code and executable scripts
├── tests/            # Unit tests
├── configs/          # Configuration files
├── Makefile          # Common tasks and commands
├── pyproject.toml    # Project configuration
└── README.md         # This file
```

## Setup

### 1. Install pyenv (if not already installed)

Follow the [pyenv installation guide](https://github.com/pyenv/pyenv#installation).

### 2. Set up virtual environment

```bash
pyenv install 3.12
pyenv virtualenv 3.12 field-extraction
pyenv local field-extraction
```

### 3. Install dependencies

```bash
make install
```

This will install:
- All project dependencies (ruff, pytest, pre-commit, etc.)
- Pre-commit hooks

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your configuration values
```

## Usage

### Code Quality

- **Linting**: `make lint`
- **Formatting**: `make format`
- **Run checks**: `make check`
- **Run tests**: `make test`

## Development

### Adding New Scripts

1. Add your script to the `scripts/` directory
2. If it requires shared code, create a submodule in `scripts/`
3. Add unit tests in `tests/` for any shared code

### Testing


Run tests with:
```bash
make test
```

## Tools

- **Ruff**: Fast Python linter and formatter
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for code quality
- **Python-dotenv**: Environment variable management

## Makefile Commands

Run `make help` to see all available commands.
