# CS573 Mushroom Project

This repository contains code for explainability analyses on mushroom classification models (Naïve Bayes, Decision Tree, Neural Network) using LIME and k‑medoids.

We used public data from the University of California, Irvine. It can be found here: https://www.kaggle.com/datasets/uciml/mushroom-classification.

Our final results can be found here: https://github.com/DerrickhSun/CS573MushroomProject/blob/main/CS573%20Mushroom%20Project.pdf.

## Getting started

### Prerequisites
- Git
- Python 3.11
- `pip` package manager

### Steps

1. Cloning the repository
```bash
git clone https://github.com/YourUser/CS573MushroomProject.git
cd CS573MushroomProject
```

2. Creating and activating a virtual environment
```bash
# Create venv
python3.11 -m venv .venv

# Activate venv (macOS/Linux)
source .venv/bin/activate
#Activate venv on gitbash
source .venv/Scripts/activate
# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

3. Installing dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running with LIME
The notebooks in the eval/lime folder are for running with LIME, and will output their results in lime_results. For example:

### Running the LIME + NBC pipeline + NBC pipeline
```bash
# Run the LIME Naïve Bayes notebook
jupyter notebook LIMENBC.ipynb
```

### Running the Decision Tree + LIME pipeline
```bash
jupyter notebook DTLIME.ipynb
```

## Running a Classification model
The classification models can also be run without the explainability analysis.

### Naive Bayes Classifier
To run the Naive Bayes classifier:
```bash
python models/nbc.py
```

### Decision Tree
To run the decision tree classifier:
```bash
python models/tree_impl.py
```
Alternatively, look at tree_impl.ipynb.

### Neural Network
To run the neural net classifier:
```bash
python models/neural_nets.py
```
The resulting neural network will be saved as "nn_CEL," which should be able to be restored with:
```bash
model.load_state_dict(torch.load("nn_CEL", weights_only=True))
```

## Ignoring the virtual environment
To prevent your `.venv/` directory from being tracked:

```bash
# Create or update .gitignore
echo ".venv/" >> .gitignore

## Committing your changes to Git
1. Check status:
   ```bash
   git status
   ```
2. Stage files:
   ```bash
   git add <files>
   git add .gitignore

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
- Use `requirements-runtime.txt` for core installs (see section on installing dependencies above)
- Follow the branch naming conventions (`nn`, `NBC`, `dt`) when pushing new code.
