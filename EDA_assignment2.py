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

for column in df.columns:
    col_mean = np.mean(df[column])
    col_dev = np.var(df[column])
    df[column] = (df[column]-col_mean)/col_dev
#scaler = preprocessing.MinMaxScaler()
#df_scaled = scaler.fit_transform(df)
#df = pd.DataFrame(df_scaled)
#df = pd.DataFrame(data=df_scaled[1:,1:],columns=df_scaled[0,1:])
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
df = df.drop(df[to_drop], axis=1)
#plt.figure(figsize=(200,100))
#sbn.heatmap(c, cmap="BrBG", annot=True)
#plt.show()
#df.to_csv('processed.csv')

input_dim = 29
output_dim = 1
hidden_dim = 150

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim))
#batch_size = 1028
#input = torch.rand(batch_size, input_dim)
#output = model(input)

#class TwoLayerNet(nn.Module):
#    def __init__(self, D_in, H, D_out):
#        super(TwoLayerNet, self).__init__()
#
#        self.linear1 = nn.Linear(D_in, H)
#        self.linear2 = nn.Linear(H, D_out)
#        self.relu = nn.ReLU()
#
#    def forward(self, x):
#        h_relu = self.relu(self.linear1(x))
#        y_pred = self.linear2(h_relu)
#        return y_pred

#N, D_in, H, D_out = 1028, 29, 159, 1
#model = TwoLayerNet(D_in, H, D_out)

df_tensor = torch.from_numpy(df.values).float()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
y_pred = model(df_tensor)
y_out = y_pred.squeeze().type(torch.FloatTensor)
y=y.type(torch.FloatTensor)
loss = loss_fn(y.unsqueeze(1), y_out)
print(loss)
