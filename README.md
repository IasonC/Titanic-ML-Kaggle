# Titanic: Machine Learning from Disaster

## Data Cleaning & Processing

The first step of the model is data visualisation and analysis with *pandas*. To predict survivability, the following features from the Kaggle dataset were used:
["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]. Sex is a binary categorical variable so it was converted to a quantitative variable by one-hot encoding. These features were selected because they were the most highly-correlated with Survivability in the training set. The "Cabin" variable was dumped because it was categorical, non-binary and missing too much data, meaning it would be hard to impute accurately without inducing noise.

Further, it was found by pandas analysis that the Age variable was missing datapoints. The data was imputed via __Multiple Imputation by Chained Equations__ (MICE), which is a robust imputation algorithm.

## Deep Neural Network (DNN) solution

__Hyperparameter-tuned DNN with AdamW optimiser (with weight decay regularisation), early stoppage, input data normalisation and holdout cross-validation.__

α = 0.00014341, λ = 0.009, layer_dims = [6, 15, 10, 5, 1].

## Random Forest Classifer (RF) solution

Random forest classifier built using sklearn. As shown in the code, the RF was tuned to the best depth and other hyperparameters using the MAE performance metric from sklearn.

## Discussion

From online reading, it seems that 80%+ performance on the test set is considered very good. The NN was slightly outperformed by the RF, which means that there is future work in identifying a better set of hyperparameters α, λ, layer_dims (and also β1, β2, ε) to outperform RF.
