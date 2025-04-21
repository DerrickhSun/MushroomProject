# CS573MushroomProject# CS573 Mushroom Project

This repository contains code for explainability analyses on mushroom classification models (Naïve Bayes, Decision Tree, Neural Network) using LIME and k‑medoids.

## Prerequisites
- Git
- Python 3.12
- `pip` package manager

## Cloning the repository
```bash
git clone https://github.com/YourUser/CS573MushroomProject.git
cd CS573MushroomProject
```

## Creating and activating a virtual environment
```bash
# Create venv
python3.12 -m venv .venv

# Activate venv (macOS/Linux)
source .venv/bin/activate

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

## Installing dependencies
```bash
pip install --upgrade pip
pip install -r requirements-runtime.txt
# If you need dev tools (Jupyter, tests):
pip install -r requirements-dev.txt
```

## Running the LIME + NBC pipeline
```bash
# Run the LIME Naïve Bayes notebook
yjupyter notebook LIMENBC.ipynb
```

## Running the Decision Tree + LIME pipeline
```bash
jupyter notebook DTLIME.ipynb
```

## Committing your changes to Git
1. Check status:
   ```bash
git status
```
2. Stage files:
   ```bash
git add <files>
# Or to add all (respecting .gitignore):
git add .
```
3. Commit:
   ```bash
git commit -m "Your descriptive commit message"
```
4. Push to your branch (e.g. `NBC` or `dt`):
   ```bash
git push --set-upstream origin <branch-name>
```

## Notes
- Ensure `.venv/` is listed in `.gitignore`.
- Use `requirements-runtime.txt` for core installs and `requirements-dev.txt` for Jupyter, tests, etc.
- Follow the branch naming conventions (`nn`, `NBC`, `dt`) when pushing new code.

