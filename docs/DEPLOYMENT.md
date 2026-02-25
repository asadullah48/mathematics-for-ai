# Deployment Guide

## ✅ Completed

- [x] Code committed to git
- [x] Changes pushed to GitHub: https://github.com/asadullah48/mathematics-for-ai

## 📦 PyPI Deployment

### Prerequisites
1. Create account at https://pypi.org
2. Generate API token from Account Settings → API tokens
3. Store token: `pip install keyring && keyring set https://upload.pypi.org/legacy/ your-username`

### Build and Publish
```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### Verify Installation
```bash
pip install mathematics-for-ai
```

## 📚 Documentation Deployment (GitHub Pages)

### Option 1: Manual Deployment
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy --force
```

### Option 2: Automated (CI/CD)
The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically deploys documentation when:
- Push to `main` branch
- Documentation builds successfully

### Enable GitHub Pages
1. Go to repository Settings → Pages
2. Source: Select "GitHub Actions"
3. The workflow will deploy to `https://asadullah48.github.io/mathematics-for-ai`

## 🔄 CI/CD Pipeline

GitHub Actions will automatically:
- Run tests on push/PR
- Check code style (flake8, black)
- Type checking (mypy)
- Build distribution packages
- Deploy documentation (on main branch push)

### View Workflow Status
https://github.com/asadullah48/mathematics-for-ai/actions

## 🚀 Quick Start for Users

```bash
# Install from PyPI (after publishing)
pip install mathematics-for-ai

# Or install from source
pip install git+https://github.com/asadullah48/mathematics-for-ai.git

# Run CLI demo
math-ai demo

# Generate sample data
math-ai generate-data -n 200

# Run tests
math-ai test
```

## 📋 Post-Deployment Checklist

- [ ] Verify PyPI package page
- [ ] Test pip installation
- [ ] Check documentation site
- [ ] Verify CI/CD workflows running
- [ ] Update README with PyPI badge
- [ ] Add package to relevant Python indexes
- [ ] Announce release (social media, forums)

## 🔧 Troubleshooting

### Build fails
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Rebuild
python -m build
```

### Upload fails
```bash
# Use verbose output
twine upload dist/* -v

# Check credentials
twine upload dist/* --verbose
```

### Documentation build fails
```bash
# Check mkdocs configuration
mkdocs build --verbose

# Test locally
mkdocs serve
```
