
# CS573 Mushroom Project: LIME + Naïve Bayes + k‑medoids

## Prerequisites
- Python **3.12**
- `git` & `pip`

## Setup
```bash
git clone https://github.com/YourUser/CS573MushroomProject.git
cd CS573MushroomProject

use
python -m venv .venv
OR use this command:
py -3.12 -m venv .venv

source .venv/bin/activate      # or `.venv\Scripts\activate`
pip install -r requirements.txt

Just note: you only have to create the venv once using that above python -m venv .venv command
but every time you open a terminal to run jupyter notebook, in that same terminal you have to do this first:
run the command below
source .venv/Scripts/activate
Optional:
# sanity check
which python   # path should end in /.venv/Scripts/python
which pip      # path ends in /.venv/Scripts/pip

If that checks out, then you can start up the jupyter notebook using the command
jupyter notebook
in your command prompt. 

python -m pip install "numpy<2" pandas scikit-learn lime plotly


python limenbc.py --input data/mushrooms.csv --model-path models/nbc.pkl
