import csv
import os
import torch
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("mushroom_dataset_enum.csv")

features = [col for col in df.columns if col != 'class' and col != 'poison' and col != 'edible' and col != "Unnamed: 0"]
target = ["poison", 'edible']


'''df_hot = df
for feature in features:
    print(feature)
    onehot = pd.get_dummies(df[feature]).add_prefix(feature+":").astype(int)
    df_hot = df_hot.join(onehot)
df_hot.drop(columns = features, inplace = True)
features = df_hot.columns.tolist()
features.remove("class")
features.remove("poison")
features.remove("edible")
features.remove("Unnamed: 0")'''

# MSE_Loss, pytorch has its own implementation, but I made it too in order to make sure there's no bug/misuse
class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        # Implementation of custom loss calculation
        diff = torch.sub(y_true, y_pred)
        return torch.mean(torch.square(diff))

class neural_network(nn.Module):
    # we initialize the neural net with a bunch of embedding layers and the actual neural network
    def __init__(self, n = 1000):
        super().__init__()
        self.embeddings = []

        # embedding layers for the categorical variables
        count = 0
        for feature in features:
            n_categories = df[feature].nunique()
            embed_dim = n_categories#min(50, (n_categories + 1) // 2)
            count+=embed_dim
            self.embeddings.append(nn.Embedding(n_categories, embed_dim))
        

        
        # the actual neural network
        self.model = nn.Sequential(
            nn.Linear(count, n),
            nn.Sigmoid(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 2),
            #nn.Softmax(dim=1)
        )
        
    # predicting the labels for the data in x
    def forward(self, x):
        embed_inputs = []
        for feature_index in range(len(features)):
            embed = self.embeddings[feature_index](x[:,feature_index])
            embed_inputs.append(embed)
            #print(embed.shape)
        #print(x[:5], torch.cat(embed_inputs, dim = 1)[:5,:10])
        x = torch.cat(embed_inputs, dim = 1)
        #print(x[0,:10])
        #print(x.shape)
        return self.model(x)


# takes in dataframe for data
def train(model, optimizer, data, loss_funct, batches = 100, batch_size = 5):

    # each epoch shuffles and goes over the data
    for epoch in range(20):
        shuffled = data.sample(frac = 1)
        x = torch.from_numpy(shuffled[features].values).int()
        y = torch.from_numpy(shuffled[target].values).float()
        running_loss = 0.0

        # i is the batch number for this epoch
        for i in range(min(batches, int(data.shape[0]/batch_size))):#range(data.shape[0]):
            inputs = x[batch_size*i : batch_size*(i+1)]
            labels = y[batch_size*i : batch_size*(i+1)]

            # here is the actual running and stochastic gradient descent
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_funct(outputs, labels)
            loss.backward()
            optimizer.step()

            # we print the loss so far
            # bit convoluted, but we print every epoch (can be changed to print more frequently)
            running_loss += loss.item()
            if i % min(batches, int(data.shape[0]/batch_size)) == min(batches, int(data.shape[0]/batch_size)) - 1:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10 :.5f}')
                print("example: ", "predicted", nn.Softmax(dim=1)(outputs[0:5].detach()), ", actual", labels[0:5],'\n')
                running_loss = 0.0
        print("------------------------------")

# function that evaluates loss
def evaluate_loss(model, data, loss_funct):
    x = torch.from_numpy(data[features].values).int()
    y = torch.from_numpy(data[target].values).float()
    pred = model(x)
    print("example:",pred[0], y[0],'\n')
    return loss_funct(pred, y)

# function that converts probabilities into predictions to evaluate accuracy
def evaluate_acc(model, data) :
    x = torch.from_numpy(data[features].values).int()
    y = torch.from_numpy(data[target].values).int()
    y_hat = nn.Softmax(dim=1)(model(x))
    pred = torch.tensor([1 if i[0] >= 0.5 else 0 for i in y_hat])
    print("sample:",y_hat[:5],y[:5])
    #print(y[:,0])
    return 1-torch.sum(torch.abs(pred-y[:,0])).item()/pred.shape[0]

# function that splits data into training, validation, and testing sets
def split(data, train_ratio = 0.8, validation_ratio = 0.1):
    shuffled = data.sample(frac = 1)
    divider1 = int(train_ratio*shuffled.shape[0])
    divider2 = int((train_ratio+validation_ratio) * shuffled.shape[0])
    train_set = shuffled[:divider1]
    valid_set = shuffled[divider1 : divider2]
    test_set = shuffled[divider2:]
    return train_set, valid_set, test_set

if __name__ == "__main__":
    #print(df[features])
    train_set, valid_set, test_set  = split(df, train_ratio = 0.9, validation_ratio = 0)
    train_set.to_csv("training.csv")
    valid_set.to_csv("validation.csv")
    test_set.to_csv("testing.csv")
    
    #train_set = train_set[train_set['poison']==1]
    #train_set.to_csv("test.csv")

    model = neural_network(n=10000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    print(f'[  base  ] loss: {evaluate_loss(model, train_set, nn.CrossEntropyLoss()) :.5f}')
    train(model, optimizer, train_set, nn.CrossEntropyLoss())
    torch.save(model.state_dict(), "nn_CEL")
    print("training:", evaluate_acc(model, train_set))
    print("test:", evaluate_acc(model, test_set))

    print("prob:", evaluate_acc(model, df))

    
    
    model.load_state_dict(torch.load("nn_CEL", weights_only=True))
    print("test:", evaluate_acc(model, df))


    '''models = []
    for i in range(1, 10, 1):
        percentile = 0.1*i
        model = quantile_regression()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

        
        train(model, optimizer, df, quantile_loss())
        models.append(model)
        torch.save(model.state_dict(), "quantile_regression"+str(int(percentile*100))+"percentile")
        print("saved percentile regression " + str(percentile))'''