### Neural Network solution built from scratch with Python

```
Training Cost after epoch 0: 0.7561181767062014
Training Cost after epoch 200: 0.6619899788775067
Training Cost after epoch 400: 0.597306843338508
Training Cost after epoch 600: 0.5448120799541024
Training Cost after epoch 800: 0.49539015531313596
Training Cost after epoch 1000: 0.45467719299893866
Training Cost after epoch 1200: 0.4218615101660319
Training Cost after epoch 1400: 0.3990955829151976
Training Cost after epoch 1600: 0.3844064425720208
Training Cost after epoch 1800: 0.37388944021130116
Training Cost after epoch 2000: 0.36598220642689316
Training Cost after epoch 2200: 0.35975287168693454
Training Cost after epoch 2400: 0.3545237269079123
Training Cost after epoch 2600: 0.3497172312371567
Training Cost after epoch 2800: 0.34580576445042244
Training Cost after epoch 3000: 0.34205462799791214
Training Cost after epoch 3200: 0.33823522328139816
Training Cost after epoch 3400: 0.3343666559658569
Training Cost after epoch 3600: 0.33019693439871356
Training Cost after epoch 3800: 0.3258756571449182
Training Cost after epoch 4000: 0.32181820071109907
Training Cost after epoch 4200: 0.317695676896284
Training Cost after epoch 4400: 0.31331492748820644
Training Cost after epoch 4600: 0.30924867188248595
Training Cost after epoch 4800: 0.3054036344997495
Training Cost after epoch 5000: 0.3017512081227985
```

The training- and dev-set accuracy for the NN is as follows:
```
Training Accuracy - 0.8701622971285893
Dev Accuracy - 0.8333333333333334
```

### Regularisation
The NN is trained with __early stoppage__ at 5000 epochs because after this point, the dev cost rises and Dev Accuracy decreases. Furthermore, this NN is trained with the Adam Optimiser with __Weight Decay__ (AdamW). Weight decay governed by the hyperparameter λ is used over the standard L2 Regularisation, because L2 regularisation does not perform as well as Weight Decay in Adam. The difference between the two regularisation methods is that AdamW updates the learned weights and biases directly while L2 updates the gradients and slightly alters the gradient update formula.

### Hyperparameter Random Search
It was determined by random logarithmic search that a good hyperparameter pair is
```
learning_rate=0.00014341, lambd=0.009
```
The other hyperparameters β1 (momentum), β2 (RMSprop) and ε (RMSprop) are set to their default values and not tuned.
