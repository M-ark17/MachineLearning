import sys 
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df = df.drop(df[to_drop], axis=1)
N, D_in, H, D_out = df.shape[0], df.shape[1], 100, 1

# Create random Tensors to hold inputs and outputs
df_tensor = torch.from_numpy(df.values).float()
x = df_tensor.squeeze(1) 

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y.long())
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
