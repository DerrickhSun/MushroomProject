import torch
import pandas as pd
import os
print(os.getcwd())
from models.neural_nets import neural_network
from models.neural_nets import evaluate_acc
import numpy as np
import matplotlib.pyplot as plt

#train_set, valid_set, test_set  = nn.split(nn.df, train_ratio = 0.9, validation_ratio = 0)
model = neural_network(n=10000)
model.load_state_dict(torch.load("nn_CEL", weights_only=True))

df = pd.read_csv("data/mushroom_dataset_enum.csv")

features = [col for col in df.columns if col != 'class' and col != 'poison' and col != 'edible' and col != "Unnamed: 0"]
target = ["poison", 'edible']
print(df[features + target])

test_set = pd.read_csv("data/testing.csv")
print("test:", evaluate_acc(model, df))