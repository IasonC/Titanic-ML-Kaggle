import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifying ML model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

titanictrain = pd.read_csv("ImpTitanicTrain.csv") # import titanic train data
    # imputed data so no NaN values
titanictest = pd.read_csv("ImpTitanicTest.csv")

# Optimise RF n_estimators and max_depth:

y = titanictrain["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"] # ML model considers these in the prediction
X = titanictrain[features] # remove Survived column

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
def get_mae(n_estimators, max_depth, train_X, train_y, val_X, val_y):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    rf.fit(train_X, train_y)
    y_pred = rf.predict(val_X)
    return mean_absolute_error(val_y, y_pred)
'''
# optimise # of trees
mae = []
n_est = [450, 475, 500, 525, 550]
for n in n_est:
    mae.append(get_mae(n, 5, train_X, train_y, val_X, val_y))

best_n = n_est[mae.index(min(mae))]

# Optimise depth
mae = []
depth = [7, 8, 9, 10, 11, 12, 13]
for d in depth:
    mae.append(get_mae(best_n, d, train_X, train_y, val_X, val_y))

best_d = depth[mae.index(min(mae))]
'''
# OPTIMISED HYPERPARAMETERS:
best_n = 500
best_d = 8

## Random Forest ML:

X_test = titanictest

model = RandomForestClassifier(n_estimators=best_n, max_depth=5, random_state=1)
    # Random Forest of 100 trees at depth 5 (max_depth)
model.fit(X, y) # fit the RF model on feature train data and labelled outcome (survived = y)
    # model trained
predictions = model.predict(X_test) # apply trained model on the test data to generate survival predictions

output = pd.DataFrame({'PassengerID' : [i+892 for i in range(418)], 'Survived' : [int(p) for p in predictions]})
output.to_csv("submission.csv", index = False)
print("Submission Saved!")