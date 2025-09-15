# Contributing to RAG Knowledge Assistant

We welcome contributions to the RAG Knowledge Assistant project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)
- Basic knowledge of FastAPI, machine learning, and RAG systems

### First Steps

1. Fork the repository on GitHub
2. Clone your fork locally:
git clone https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git
cd rag-knowledge-assistant

text

3. Set up the development environment:
.\scripts\setup.ps1

text

4. Create a new branch for your contribution:
git checkout -b feature/your-feature-name

text

## Development Setup

### Environment Configuration

1. Copy the environment template:
cp .env.example .env

text

2. Add your API keys and configuration to `.env`:
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key

Add other required keys
text

3. Install dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm

text

### Running the Development Server

.\scripts\dev.ps1

text

The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

### Using Docker for Development

.\scripts\docker.ps1

text

## Contributing Process

### 1. Issue Tracking

- Check existing issues before creating new ones
- Use issue templates when available
- Clearly describe the problem or enhancement request
- Add appropriate labels and assign yourself if working on it

### 2. Pull Request Process

1. **Create a feature branch**:
git checkout -b feature/description-of-feature

text

2. **Make your changes**:
- Follow coding standards
- Add tests for new functionality
- Update documentation as needed

3. **Test your changes**:
.\scripts\test.ps1

text

4. **Commit your changes**:
git add .
git commit -m "feat: add new feature description"

text

5. **Push to your fork**:
git push origin feature/description-of-feature

text

6. **Create a Pull Request**:
- Use the PR template
- Provide clear description of changes
- Link related issues
- Request review from maintainers

### 3. Commit Message Convention

We use conventional commits for clear commit history:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
feat: add support for Anthropic Claude models
fix: resolve memory leak in vector store
docs: update API documentation for query endpoint
test: add integration tests for document upload

text

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use `isort` for import sorting
- **Formatting**: Use `black` for code formatting
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Code Formatting

Before committing, ensure your code is properly formatted:

Format code
black backend/

Sort imports
isort backend/

Check linting
flake8 backend/

text

### Type Hints

All public functions must include type hints:

from typing import List, Optional, Dict, Any

def process_documents(
documents: List[str],
batch_size: int = 32,
metadata: Optional[Dict[str, Any