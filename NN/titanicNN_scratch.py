"""
Created on Wed Jul 20 21:43:17 2022

Author: Iason Chaimalas

Deep NN Binary Classifier (ReLU hidden activation; Sigmoid output activation)
Optimisation Algo: Adam
Batch Gradient Descent (m = 891 so too few for Mini-Batch GD)

"""

import numpy as np # vectorised NN for quick processing
import pandas as pd # to read in the data stored in pd dfs
import matplotlib.pyplot as plt # for plotting the cost function
from tnn_utils import *

titanictrain = pd.read_csv("ImpTitanicTrain.csv") # import titanic train data
    # imputed data so no NaN values
titanictest = pd.read_csv("ImpTitanicTest.csv")

# Train Set
Y = np.array(titanictrain["Survived"]).reshape(1,-1)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"] # ML model considers these in the prediction
X = np.transpose(np.array(titanictrain[features]))
    # remove Survived column, convert to np array and transpose
    # now 891 training examples are stacked horizontally (each example is a column of X)

# Test Set
test_X = np.transpose(np.array(titanictest[features]))
'''
print(train_X.shape)
print(dev_X.shape)
print(train_Y.shape)
print(dev_Y.shape)
print(test_X.shape)
'''
def train_dev_split(X, Y, seed, split=0.8):

    assert(0 < split < 1)

    # Train-Dev Split
    np.random.seed(seed) # reproducibility
    m = X.shape[1] # number of training examples

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    train_X = shuffled_X[:, 0:int(m*split)]
    dev_X = shuffled_X[:, int(m*split):]
    train_Y = shuffled_Y[:, 0:int(m*split)]
    dev_Y = shuffled_Y[:, int(m*split):]

    return train_X, dev_X, train_Y, dev_Y

def normalise_input(X):
    """
    Normalise the input data to have mean 0 and standard deviation 1.

    Inputs: X - training set
    Outputs: X - normalised training set
    """

    for i in range(X.shape[0]):
        X[i,:] = (X[i,:] - np.mean(X[i,:])) / np.std(X[i,:])
    
    return X

def initialise_parameters(layer_dims):
    """
    Parameter initialisation for the NN

    Inputs: layer_dims - list of the dimensions of each layer in the NN
    Outputs: parameters - dictionary of the parameters of the NN
        W1, b1, ... WL, bL - the weight matrix & bias vector parameters of the layers

    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # number of NN layers

    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1], dtype=np.float64)
            # He initialisation - optimal for ReLU hidden units
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1), dtype=np.float64) # b initialised to zeros
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialise_adam(parameters):
    """
    Initialise v, s for Adam optimisation algorithm.

    v = exponentially-weighted average for gradient (velocity -> used in GD with Momentum)
    s = exponentially-weighted average for squared gradient (used in RMSprop)

    s + v are together used in Adam. Adam = Momentum + RMSprop
    
    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape, dtype=np.float64)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape, dtype=np.float64)
        s["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape, dtype=np.float64)
        s["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape, dtype=np.float64)

    return v, s

def compute_cost(AL, Y):
    """
    Compute model cost (cost func = average cross-entropy cost)

    Inputs: AL - probability vector of output layer
            Y - true labels vector
    Outputs: model cost

    """

    m = Y.shape[1]

    cost = np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)) / -m

    #loss = np.multiply(Y, -np.log(AL)) + np.multiply(1-Y, -np.log(1-AL))
    #cost = 1./m * np.sum(loss)
    cost = np.squeeze(cost) # to force unidimensionality
    
    return cost

def single_forward(A_prev, W, b, activation):
    #print("A_prev :", A_prev.shape)
    #print("W :", W.shape)

    Z = np.dot(W, A_prev) + b
    single_cache = (A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (single_cache, activation_cache)

    return A, cache

def forward_propagation(X, parameters):
    """
    Forward propagation through the NN and loss-computation per epoch

    """

    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the NN
        # -- div by 2 since params contains W,b per layer

    for l in range(1,L):
        A_prev = A
        A, cache = single_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = single_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    #AL[AL == 1] = 0.999999 # to avoid overflow

    return AL, caches

def single_backward(dA, cache, activation):
    """ Backpropagation for a single lth layer """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    """ Backpropagation through the L-layer NN """

    grads = {} # hold gradieunts to be updated per epoch
    L = len(caches) # number of NN layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # make sure Y shape is same as AL

    # initialise backprop
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost w.r.t. AL

    # Lth layer
    current_cache = caches[-1] # start at last layer -> take last cache current
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = single_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = single_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = dA_prev_temp, dW_temp, db_temp

    return grads

def update_params(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, lambd=0.01):
    """
    Update learned parameters W, b via Adam Gradient Descent
    
    v,s are parameters for Adam optimisation algorithm.
    t is the epoch number used in bias correction
    Learning rate α, β1, β2, ε are hyperparameters
    
    """

    L = len(parameters) // 2 # number of layers in the NN
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        # moving avg of gradients
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # bias correction for moving avg of gradients
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # moving avg of squared gradients
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)

        # bias correction for moving avg of squared gradients
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # update parameters
        parameters["W" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] -= learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)

        # weight decay -> AdamW
        parameters["W" + str(l+1)] -= learning_rate * lambd * parameters["W" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * lambd * parameters["b" + str(l+1)]

    return parameters, v, s

def model(X, Y, layer_dims, learning_rate=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, lambd=0.01, epochs_num=30000, print_cost=True):
    """
    3-layer neural network model: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    AdamW optimiser with β1, β2, ε, λ hyperparameters (λ is weight decay regularisation).
    Learning rate α used for gradient descent.
    Training over epochs_num epochs.
    Print cost per 2000 epochs if print_cost is True.

    """

    seed = 1 # for reproducibility
    train_costs = [] # store cost per epoch on training set
    dev_costs = [] # store cost per epoch on dev set

    # initialise parameters
    parameters = initialise_parameters(layer_dims) # initialise parameters
    v, s = initialise_adam(parameters) # initialise Adam optimiser parameters

    for t in range(epochs_num + 1):

        if t % 1000 == 0:
            # new train-dev 80-20 split every 10000 epochs
            train_X, dev_X, train_Y, dev_Y = train_dev_split(X, Y, seed)
            #seed += 1 # increment to get new split next time

            train_X = normalise_input(train_X)
            dev_X = normalise_input(dev_X)

        # TRAINING SET
        train_AL, train_caches = forward_propagation(train_X, parameters) # forward propagation

        train_AL[train_AL == 1] = 0.99 # to avoid overflow

        train_cost = compute_cost(train_AL, train_Y) # compute cost
        grads = backward_propagation(train_AL, train_Y, train_caches) # backpropagation
        parameters, v, s = update_params(parameters, grads, v, s, t+1, learning_rate, beta1, beta2, epsilon, lambd) 
            # update parameters

        # DEV SET
        dev_AL, dev_caches = forward_propagation(dev_X, parameters) # forward propagation
        dev_AL[dev_AL == 1] = 0.99 # to avoid overflow
        dev_cost = compute_cost(dev_AL, dev_Y) # compute cost
        
        if print_cost and t % 1000 == 0:
            print(f"Training Cost after epoch {t}: {train_cost}")
            train_costs.append(train_cost)
            dev_costs.append(dev_cost)

    if print_cost:
        # plot the cost
        plt.plot(train_costs)
        plt.plot(dev_costs)
        plt.ylabel('Cost J')
        plt.xlabel('Epochs (per 1000)')
        plt.title(f"Learning rate = {learning_rate}")
        plt.legend(['Train', 'Dev'])
        plt.show()
        plt.savefig("cost_plot.png")

    return parameters

def predict(X, Y, parameters):
    """
    Make predictions with the trained model (learned parameters W,b) 
    on X and evaluate accuracy against true Y labels.
    
    """

    m = X.shape[1]
    L = len(parameters) // 2 # number of layers in the NN

    X = normalise_input(X) # normalise input
    AL, caches = forward_propagation(X, parameters)

    # convert to hardmax predictions on Survival
    AL[AL > 0.5] = 1
    AL[AL <= 0.5] = 0

    accuracy = np.sum((AL == Y))/m

    return AL, accuracy

def predict_test(X, parameters):
    """ Save predictions on test set to submit on Kaggle """

    X = normalise_input(X)
    predictions, caches = forward_propagation(X, parameters)

    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    output = pd.DataFrame({'PassengerID' : [i+892 for i in range(418)],
                           'Survived' : list(map(lambda x: int(x), predictions.flatten()))})
    output.to_csv("submission.csv", index = False)
    print("Submission Saved!")

def hyperparameter_search(X, Y, layer_dims):
    """ Hyperparameter random search """

    r = -(2*np.random.rand(15)+2) # α set to 5 random values between 0.01 and 0.0001
    n = -(2*np.random.rand(15)+1) # λ set to 5 random values between 0.1 and 0.001
    learning_rates = np.power(10, r)
    lambdas = np.power(10, n)
    ta = [] # training accuracy for model trained on each α
    va = [] # dev accuracy for model trained on each α

    train_X, val_X, train_Y, val_Y = train_dev_split(X, Y, seed=100, split=0.9) # 90-10 train-val split
        # validation set not used in model training -> holdout cross validation

    for a in learning_rates:
        for l in lambdas:
            parameters = model(train_X, train_Y, layer_dims, learning_rate=a, lambd=l, epochs_num=100000, print_cost=False)

            predictions_train, train_accuracy = predict(train_X, train_Y, parameters)
            predictions_val, val_accuracy = predict(val_X, val_Y, parameters)

            ta.append(train_accuracy)
            va.append(val_accuracy)

            print(f"α = {a}, λ = {l}")

    '''
    best_learning_rate_train = learning_rates[ta.index(max(ta))]
    best_learning_rate_dev = learning_rates[va.index(max(va))]
    best_lambd_train = lambdas[ta.index(max(ta))]
    best_lambd_dev = lambdas[va.index(max(va))]

    print(f"Best hyperparameters for training set: α = {best_learning_rate_train}, λ = {best_lambd_train}")
    print(max(ta))
    print(ta.index(max(ta)))
    print(f"Best hyperparameters for dev set: α = {best_learning_rate_dev}, λ = {best_lambd_dev}")
    print(max(da))
    print(va.index(max(va)))
    '''
    print("\n")
    print("ta:", ta)
    print("va:", va)
    print("α:", learning_rates)
    print("λ:", lambdas)

    ## -> best learning rate for training set: 0.004965119020953387

# HOLDOUT VALIDATION SET
train_X, val_X, train_Y, val_Y = train_dev_split(X, Y, seed=100, split=0.9) # 90-10 train-val split
    # val set held aside for validation of model performance
layer_dims = [train_X.shape[0], 15, 10, 5, 1] #[train_X.shape[0], 15, 10, 5, 1]

# hyperparameter_search(X, Y, layer_dims) # -> best_a = 0.00014341, best_l = 0.01290933 - 24/07/2022
# hyperparameter_search(X, Y, layer_dims) # -> best_a = 0.009647072660594253, best_l = 0.01465328 - 23/07/2022
# hyperparameter_search(X, Y, layer_dims) # -> lambd = 0.02647225, learning_rate = 0.0093066 - 22/07/2022

#parameters = model(X, Y, layer_dims, learning_rate=0.0093066, lambd=0.02647225, epochs_num=30000, print_cost=True) # - BEST SO FAR - 23/07/2022
parameters = model(train_X, train_Y, layer_dims, learning_rate=0.00014341, lambd=0.009, epochs_num=5000, print_cost=True)
#parameters = model(X, Y, layer_dims, learning_rate=1e-4, epochs_num=300000, print_cost=True)

predictions_train, train_accuracy = predict(train_X, train_Y, parameters)
predictions_dev, val_accuracy = predict(val_X, val_Y, parameters)

print(train_accuracy)
print(val_accuracy)

predict_test(test_X, parameters)