import sys 

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn

sbn.set(color_codes=True)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
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

#for column in df.columns:
#    col_mean = np.mean(df[column])
#    col_dev = np.var(df[column])
#    df[column] = (df[column]-col_mean)/col_dev
scaler = preprocessing.MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled)
df = pd.DataFrame(data=df_scaled[1:,1:],columns=df_scaled[0,1:])
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(df[to_drop], axis=1)
#plt.figure(figsize=(200,100))
#sbn.heatmap(c, cmap="BrBG", annot=True)
#plt.show()
#df.to_csv('processed.csv')

input_dim = df.shape[1]
output_dim = 1
hidden_dim = 150

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
model.apply(init_weights)
df_tensor = torch.from_numpy(df.values).float()
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.requires_grad = True
for param in model.parameters():
    param.requires_grad = True
learning_rate = 1e-4
max_epochs = 100
batch_size = df.shape[0]
n_batches = 32
batch_size = 1028
biter_size = int(batch_size/n_batches)
print(model)

for epoch in range(max_epochs):
    for i in range(biter_size):
        # Local batches and labels
        local_X = df_tensor[i*n_batches:(i+1)*n_batches,:]
        local_Y = y[i*n_batches:(i+1)*n_batches]
        y_pred = model(local_X)
        y_out = y_pred.squeeze().type(torch.FloatTensor)
        y=y.type(torch.FloatTensor)
        loss = loss_fn(local_Y.unsqueeze(1), y_out.long())
        print(loss)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
                
