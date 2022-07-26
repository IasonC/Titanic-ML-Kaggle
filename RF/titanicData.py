import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.experimental import enable_iterative_imputer # enable MICE imputer
from sklearn.impute import IterativeImputer # MICE imputer

titanic = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Show first & last 5 rows and dimensions of the 
#print(titanic.head())
#print(titanic.tail())
#print(titanic.shape)

# Info on features, their non-null values and entry type
print(titanic.info())

print(titanic.isna().sum())
    # 177 NaN age, 687 NaN Cabin, 2 NaN Embarked
    # other 8 features are fully filled

# get rows with NaN empty values for age, cabin, embarked (the sections with missing vals)
nullsage = titanic[titanic["Age"].isna()]
nullscabin = titanic[titanic["Cabin"].isna()]
nullsembarked = titanic[titanic["Embarked"].isna()]

# get list of indices for people missing age, cabin, embarked data
ageid = list(nullsage.index)
cabid = list(nullscabin.index)
embid = list(nullsembarked.index)

del titanic["Cabin"]
    # too much missing data and categorical - really hard to impute - dump

for i in range(titanic.shape[0]):
    if titanic.iloc[i, 4] == "male": titanic.iloc[i, 4] = 1
    else: titanic.iloc[i, 4] = 0
    # "Sex" is 4th column.
    # one-hot encoding of Sex as 1/0 male/female in TRAIN DATA

for i in range(test.shape[0]):
    if test.iloc[i, 3] == "male":
        test.iloc[i, 3] = 1
    else:
        test.iloc[i, 3] = 0
    # one-hot gender encoding in TEST DATA

titanic_corr = titanic.corr()["Survived"]
    # high neg correlation with Pclass and high pos corr with Fare

""" MICE for Train Data """

mX = titanic.drop(titanic.columns[[0, 3, 8, 10]], axis = 1) # keep only numerical vars
print(mX)
imp = IterativeImputer() # imputer trains BayesianRidge() regression model
mX = imp.fit_transform(mX) # MICE imputation
titanic_imp = pd.DataFrame(mX) # mX is np array -> conv back to dataframe
print(titanic_imp)

titanic_imp.to_csv("ImpTitanicTrain.csv", index = False)
    # columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

""" MICE for Test Data """

testX = test.drop(test.columns[[0, 2, 7, 9, 10]], axis = 1) # drop unused features
testimp = IterativeImputer()
testX = testimp.fit_transform(testX)
test_imp = pd.DataFrame(testX)

test_imp.to_csv("ImpTitanicTest.csv", index = False)
    # columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]