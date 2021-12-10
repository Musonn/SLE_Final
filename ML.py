#Julian Brown
#SLE
#Final Project
#Machine Learning

import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


#%% Data Processing

#Open csv file and read data
with open('pergame.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    data = []
    for row in csv_reader:
        data.append(row)

data = np.array(data)

statspergame = data[1:,3:].astype(float)
X_train = statspergame[:,:-1]
y_train = statspergame[:,-1]/16

#Open csv file and read data
with open('2021pergame.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    data = []
    for row in csv_reader:
        data.append(row)

data = np.array(data)

pergame2021 = data[1:,2:].astype(float)
numgames = pergame2021[:,0]

for i in range(pergame2021[:,0].size):
    pergame2021[i,-1] = pergame2021[i,-1]/pergame2021[i,0]
    
pergame2021 = pergame2021[:,1:]
X_test = pergame2021[:,:-1]
y_test = pergame2021[:,-1]

scaler = MinMaxScaler()
scaledtrain = scaler.fit_transform(X_train)
scaledtest = scaler.transform(X_test)


#%%Ridge Regression

ridge = RidgeCV(alphas=[1e-2,1e-1, 1,10,20]).fit(scaledtrain, y_train)
guessridge = ridge.predict(scaledtest)

print('Ridge Regression:')
print('R^2 = ', round(ridge.score(scaledtest, y_test),3))

#%% Logistic Regression
#Other parameters were tried to attempt to improve model but they resulted in warnings

param_grid = {'C': [10, 1, 0.1, 0.001],
              'penalty': ['l2', 'none'],
              'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']}
 
#log_grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 1)
 
log_grid = LogisticRegression()
# fitting the model for grid search
#Win rate had to be converted to number of wins as sklearn
#requires target to be int
log_grid.fit(scaledtrain, (y_train*16).astype(int))

# print(log_grid.best_params_)
# print(log_grid.best_estimator_)

log_grid_predict = log_grid.predict(scaledtest)
logscore = log_grid.score(scaledtest, np.multiply(numgames,y_test))
#Once again convert target to int by multiplying by games played
print('Logistic Regression: ')
print('R^2 = ', round(logscore,3))

#%%Support Vector Regression

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
 
svr_grid = GridSearchCV(SVR(), param_grid, scoring = 'r2', refit = True, verbose = 0)
 
# fitting the model for grid search
svr_grid.fit(scaledtrain, y_train)

svr_grid_predict = svr_grid.predict(scaledtest)
print('Support Vector Regression: ')
print('R^2 = ', round(svr_grid.score(scaledtest, y_test),3))