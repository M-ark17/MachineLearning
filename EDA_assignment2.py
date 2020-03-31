import sys 

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch.autograd import Variable

n = len(sys.argv)

if(n == 3):
    training_file = str(sys.argv[1])
    testing_file = str(sys.argv[2])
elif(n == 2):
    training_file = str(sys.argv[1])
else:
    training_file = "train.csv"

df = pd.read_csv("train.csv")
y = torch.from_numpy(df['Attrition'].values).float()
y = y[1:]
df = df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'ID'], axis = 1)
#df = pd.DataFrame(df)
#df = pd.DataFrame(data=df[1:,1:],columns=df[0,1:])
df = df.drop_duplicates()
df = df.dropna()
df.loc[df.Gender == 'Male', 'Gender'] = 1
df.loc[df.Gender == 'Female', 'Gender'] = -1
df.loc[df.BusinessTravel == 'Non-Travel', 'BusinessTravel'] = 1
df.loc[df.BusinessTravel == 'Travel_Rarely', 'BusinessTravel'] = 2
df.loc[df.BusinessTravel == 'Travel_Frequently', 'BusinessTravel'] = 3
df.loc[df.Department == 'Research & Development', 'Department'] = 1
df.loc[df.Department == 'Sales', 'Department'] = 2
df.loc[df.Department == 'Human Resources', 'Department'] = 3
df.loc[df.OverTime == 'Yes', 'OverTime'] = 1
df.loc[df.OverTime == 'No', 'OverTime'] = -1
df.loc[df.MaritalStatus == 'Married', 'MaritalStatus'] = 1
df.loc[df.MaritalStatus == 'Single', 'MaritalStatus'] = 2
df.loc[df.MaritalStatus == 'Divorced', 'MaritalStatus'] = 3
df.loc[df.EducationField == 'Other', 'EducationField'] = 1
df.loc[df.EducationField == 'Life Sciences', 'EducationField'] = 2
df.loc[df.EducationField == 'Medical', 'EducationField'] = 3
df.loc[df.EducationField == 'Marketing', 'EducationField'] = 4
df.loc[df.EducationField == 'Technical Degree', 'EducationField'] = 5
df.loc[df.EducationField == 'Human Resources', 'EducationField'] = 6
df.loc[df.JobRole == 'Sales Executive', 'JobRole'] = 1
df.loc[df.JobRole == 'Research Scientist', 'JobRole'] = 2
df.loc[df.JobRole == 'Laboratory Technician', 'JobRole'] = 3
df.loc[df.JobRole == 'Healthcare Representative', 'JobRole'] = 4
df.loc[df.JobRole == 'Manufacturing Director', 'JobRole'] = 5
df.loc[df.JobRole == 'Sales Representative', 'JobRole'] = 6
df.loc[df.JobRole == 'Research Director', 'JobRole'] = 7
df.loc[df.JobRole == 'Manager', 'JobRole'] = 8
df.loc[df.JobRole == 'Human Resources', 'JobRole'] = 9

scaler = preprocessing.MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled)
df = pd.DataFrame(data=df_scaled[1:,1:],columns=df_scaled[0,1:])

input_dim = df.shape[1]
output_dim = 1
hidden_dim = 200

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
    nn.Sigmoid())

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
model.apply(init_weights)
df_tensor = torch.from_numpy(df.values).float()
loss_fn = torch.nn.BCELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
max_epochs = 5
batch_size = df.shape[0]
n_batches = 32
batch_size = 1028
biter_size = int(batch_size/n_batches)
print(model)
y=y.type(torch.FloatTensor)
max_loss = 10000
for epoch in range(max_epochs):
    df_tensor = df_tensor[torch.randperm(df_tensor.size()[0])]
    running_loss = 0
    for i in range(biter_size):
        # Local batches and labels
        local_X = df_tensor[i*n_batches:(i+1)*n_batches,:]
        local_Y = y[i*n_batches:(i+1)*n_batches]
        y_pred = model(local_X)
        y_out = y_pred.squeeze().type(torch.FloatTensor)
        y_out = Variable(y_out, requires_grad = False) 
        loss = loss_fn(local_Y.unsqueeze(1), y_out)
        loss = Variable(loss, requires_grad = True)
        print(loss.data)
        running_loss += loss.detach()
        optimizer.zero_grad()
        loss.detach()
        loss.backward()
        optimizer.step()
    if(running_loss < max_loss):
        torch.save(model, "./better_model")
        max_loss = running_loss
    print("="*50)
    print("Loss for the epoch ", epoch)
    print(running_loss)
    print("="*50)
                
#print("*"*100)
#print("average Loss = ", running_loss/(max_epochs*batch_size)) 
#print("*"*100)
#from skorch import NeuralNetRegressor
#net = NeuralNetRegressor(model
#                         , max_epochs=max_epochs
#                         , lr=learning_rate
#                         , verbose=1)	
#
#param_grid = {'max_epochs': [10,20,30,100,200,300,500,1000,2000], 'lr': [1e-2,1e-4,1e-6,1e-1]}
#grid = GridSearchCV(estimator=net, param_grid=param_grid, n_jobs=-1, cv=3)
#y = y.reshape(-1, 1)
#grid_result = grid.fit(df, y)
#
#nn_opt = grid_result.best_estimator_
#
#print('='*20)
#print("best params: " + str(grid.best_estimator_))
#print("best params: " + str(grid.best_params_))
#print('best score:', grid.best_score_)
#print('='*20)
