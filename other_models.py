import sys 

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import svm, tree, linear_model, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Common sklearn Model Helpers
from sklearn import model_selection
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss

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
def EDA(filename ):
    df = pd.read_csv(filename)
    if "train" in filename:
        y = torch.from_numpy(df['Attrition'].values).float()
        y = y[1:]
        y = y.numpy()
        df = df.drop(['Attrition'], axis = 1)
        flag = 1
    else:
        flag = 0
        
    df = df.drop([ 'EmployeeCount', 'EmployeeNumber', 'ID'], axis = 1)
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
    saved_cols = df.columns
    scaler = preprocessing.MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled)
    df = pd.DataFrame(data=df_scaled[1:,1:],columns=df_scaled[0,1:])
    #df = pd.DataFrame(df,saved_cols)
    #if(flag):
    #    corr_matrix = df.corr().abs()
    #    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #    global to_drop
    #    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    #    
    #df = df.drop(df[to_drop], axis=1)
    #df = pd.DataFrame(data=df[1:,1:],columns=df_scaled[0,1:])
    #plt.figure(figsize=(200,100))
    #sbn.heatmap(c, cmap="BrBG", annot=True)
    #plt.show()
    #df.to_csv('processed.csv')
    
    #df = torch.from_numpy(df.values).float()
    if(flag):
        return df,y
    else:
        return df

df,y = EDA("train.csv")
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7)))
models.append(('Random Forest', RandomForestClassifier(n_estimators = 100, random_state = 7)))
models.append(('SVM', SVC(gamma='auto', kernel = 'rbf' ,random_state=7)))


for name, model in models:
              kfold = model_selection.KFold(n_splits=2, random_state=7)  # 10-fold cross-validation
              cv_acc_results = model_selection.cross_val_score(model, df, y, cv=kfold, scoring='accuracy')
              print(cv_acc_results.mean())


param_grid = {'C': np.arange(1e-04, 2, 0.01)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear', # setting GridSearchCV
                                         class_weight="balanced", 
                                         random_state=7),
                      iid=True,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=2)

log_grid = log_gs.fit(df, y)
log_opt = log_grid.best_estimator_
results = log_gs.cv_results_

print('='*20)
print("best params: " + str(log_gs.best_estimator_))
print("best params: " + str(log_gs.best_params_))
print('best score:', log_gs.best_score_)
print('='*20)
x_test = EDA("test.csv")
log_opt.fit(df,y)
y_test = log_opt.predict(x_test)
np.savetxt("test_prediciton_logistic.csv",[np.array(pd.read_csv('test.csv')['ID'][1::]), y_test.astype(int)],header="ID,Attrition")


rf_classifier = RandomForestClassifier(class_weight = "balanced",
                                       random_state=7)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
              'min_samples_split':[2,4,6,8,10],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_depth': [5, 10, 15, 20, 25]}

grid_obj = GridSearchCV(rf_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=2)

grid_fit = grid_obj.fit(df, y)
rf_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)
rf_opt.fit(df, y)
y_test = rf_opt.predict(x_test)

np.savetxt("test_prediciton_random_forest.csv",[np.array(pd.read_csv('test.csv')['ID'][1::]), y_test.astype(int)],header="ID,Attrition")

svm_classifier = SVC(gamma='auto', kernel = 'rbf' ,random_state=7)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
              'gamma' : [0.001, 0.01, 0.1, 1]}

grid_obj = GridSearchCV(svm_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=2)

grid_fit = grid_obj.fit(df, y)
svm_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)
svm_opt.fit(df, y)
y_test = svm_opt.predict(x_test)

np.savetxt("test_prediciton_svm.csv",[np.array(pd.read_csv('test.csv')['ID'][1::]), y_test.astype(int)],header="ID,Attrition")
