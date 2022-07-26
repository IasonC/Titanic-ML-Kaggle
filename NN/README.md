### 4-Layer Neural Network solution built from scratch with Python

### Layers
The NN is shaped as [6, 15, 10, 5, 1]. Hidden layers have ReLU activations and the output layer has a Sigmoid activation since this model is a binary classifier of passenger survival. There was a variation in model performance with more or less layers and neurons, but I empirically found that 4 layers works well for this dataset, and increasing to 5 layers decreases performance on the test and val sets. There is future work in more thoroughly testing various layer-neuron architectures to find the best-sized NN.

### Regularisation
The NN is trained with __early stoppage__ at 10000 epochs with cross-validation and 5000 epochs without cross-val because after this point, the validation cost rises and Val Accuracy decreases. Furthermore, this NN is trained with the Adam Optimiser with __Weight Decay__ (AdamW). Weight decay governed by the hyperparameter λ is used over the standard L2 Regularisation, because L2 regularisation does not perform as well as Weight Decay in Adam. The difference between the two regularisation methods is that AdamW updates the learned weights and biases directly while L2 updates the gradients and slightly alters the gradient update formula.

### Hyperparameter Random Search
It was determined by random logarithmic search that a good hyperparameter pair is
```
learning_rate=0.00014341, lambd=0.009
```
The other hyperparameters β1 (momentum), β2 (RMSprop) and ε (RMSprop) are set to their default values and not tuned.

### Normalisation
Input training data to the model was normalised by Z-standardisation with mean and standard deviation (Z = (x-μ)/σ), in order to remove data variation and equalise difference in scale between the features. Further, the test data was normalised before making predictions for the submission.csv

### Cost plot and Cost decrease
The cost plot for the NN __with cross-validation__ is as follows:
![cost_plot_crossval](https://user-images.githubusercontent.com/73920832/180900942-fae74bc9-1b4e-47c1-b538-4190a62a41fa.png)

The terminal cost per 1000 epochs with cross-val is shown:
```
Training Cost after epoch 0: 0.7561181767062014
Training Cost after epoch 1000: 0.4805551645663465
Training Cost after epoch 2000: 0.3842755082774036
Training Cost after epoch 3000: 0.37247016028689456
Training Cost after epoch 4000: 0.3655897293227129
Training Cost after epoch 5000: 0.36246756938466146
Training Cost after epoch 6000: 0.3335860920713467
Training Cost after epoch 7000: 0.3412132670026569
Training Cost after epoch 8000: 0.3214487727688435
Training Cost after epoch 9000: 0.31409249190682287
Training Cost after epoch 10000: 0.3052417766157508
```

The cost plot for the NN __without cross-validation__ is as follows:
![cost_plot_withoutcrossval](https://user-images.githubusercontent.com/73920832/180900954-e9d5c414-7646-46bf-81e6-14c22b956d92.png)

Both plots follow the expected trend for test and dev set cost.

Further, the terminal cost per 1000 epochs without cross-val is recorded below:
```
Training Cost after epoch 0: 0.7561181767062014
Training Cost after epoch 1000: 0.45467719299893866
Training Cost after epoch 2000: 0.36598220642689316
Training Cost after epoch 3000: 0.34205462799791214
Training Cost after epoch 4000: 0.32181820071109907
Training Cost after epoch 5000: 0.3017512081227985
```

### Accuracy

The NN was trained with and without holdout cross-validation. With cross-validation, the training data was shuffled and partitioned into training and validation sets (90-10 split). Then, every 200 epochs (found to be optimal), the training set was partitioned randomly (with incrementing seed to yield a new and reproducible split) into training and dev sets (80-20). The model was trained on the training set and evaluated on both the training and dev sets to yield the training-dev cost plots. Further, the final model was evaluated on the validation set which was held out from the training and thus induced no training bias. The validation accuracy was optimimsed by the hyperparameter tuning.

Without cross-validation, 80% of the training set was used in training and 20% using as the validation set (random partition at known seed). There was no reshuffling of the training set per set number of epochs.

The training- and val-set accuracy for the NN with holdout cross-validation is as follows:
```
Training Accuracy - 0.8764044943820225
Validation Accuracy - 0.8444444444444444
```
Without cross-validation:
```
Training Accuracy - 0.8701622971285893
Validation Accuracy - 0.8333333333333334
```

While the cross-validation model performs slightly better, this is not a significant performance increase. Both models perform the same on the test set (77.03%).
