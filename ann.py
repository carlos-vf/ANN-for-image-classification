# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:02:50 2022

@author: Carlos Velázquez Fernández

    Artificial Neural Networks
        (hand-written number classification)
"""

import numpy as np
import matplotlib.pyplot as plt


# Global variables
TRAINING_SET = 0.7
VALIDATION_SET = 0.1
TESTING_SET = 0.2

EPOCHS = 1000
LEARNING_RATE = 0.02
NEURONS_HIDDEN = 200


def main(file): 
    
    # Load data from .csv
    print('Loading data ...\n')
    data = np.loadtxt(file, delimiter=",")
    print('Data loaded succesfully\n')
    np.random.shuffle(data)
        
    # Separate in sets
    cutLearning = int(len(data)*(TRAINING_SET+VALIDATION_SET))
    cutValidation = int(len(data)*(TRAINING_SET))
    learning = data[:cutLearning]
    testing = data[cutLearning:]
    validation = data[cutValidation:cutLearning]
    
    # Validation data
    X_validation = validation[:,1:].T
    y = validation[:,:1]
    Y_validation = np.zeros((y.shape[0], 10))
    for i in range(len(y)):
        Y_validation[i, int(y[i])] = 1.0
    Y_validation = Y_validation.T
    
    
    # Training data
    X_learning = learning[:,1:].T
    y = learning[:,:1]
    Y_learning = np.zeros((y.shape[0], 10))
    for i in range(len(y)):
        Y_learning[i, int(y[i])] = 1.0
    Y_learning = Y_learning.T
        
    # Testing data
    X_testing = testing[:,1:].T
    y = testing[:,:1]
    Y_testing = np.zeros((y.shape[0], 10))
    for i in range(len(y)):
        Y_testing[i, int(y[i])] = 1.0  
    Y_testing = Y_testing.T



    # Initialize paramters
    neurons_input = X_learning.shape[0]
    neurons_output = Y_learning.shape[0]
    accuracyList = []
    
    w1, b1, w2, b2 = initializeParameters(neurons_input, neurons_output)

    

    # Training
    for i in range(EPOCHS):
        
        # Forward propagation
        z1, a1, z2, a2 = forward(X_learning, w1, b1, w2, b2)
        
        # Backward propagation
        dw1, db1, dw2, db2 = backward(X_learning, Y_learning, w1, b1, w2, b2, z1, a1, z2, a2)
        
        # Update parameters
        w1, b1, w2, b2 = updateParameters(w1, b1, w2, b2, dw1, db1, dw2, db2)
        
        
        accuracyList.append(accuracy(X_validation, Y_validation, w1, b1, w2, b2))
        
        if(i % 100 == 0):
            print("Accuracy in iteration", i, ":", round(accuracyList[i],2), "%")
            
            
    
    # Accuracy calculation
    print("\nAccuracy of Learning Dataset: ", round(accuracy(X_learning, Y_learning, w1, b1, w2, b2), 2), "%")
    print("\nAccuracy of Test Dataset: ", round(accuracy(X_testing, Y_testing, w1, b1, w2, b2),2), "%")
    print("\nAccuracy per class: ") 
    acc =   accuracyPerClass(X_testing, Y_testing, w1, b1, w2, b2)     
    for i in range(acc.shape[0]):
        print("Accuracy of ", i, ":", round(acc[i], 2), "%")

    # Plot
    x = [i for i in range(0, EPOCHS)]
    plt.plot(x, accuracyList, linewidth=2.0)
    plt.xlim(0, EPOCHS)
    plt.ylim(0, 100)
    plt.title("Evolution of accuracy in the validation set")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")
    
    


# w1: matrix of size (NEURONS_HIDDEN, neurons_input) with values in range [-1, 1]. WEIGHTS1
# b1: matrix of size (NEURONS_HIDDEN, 1) of zeros. BIAS1
# w2: matrix of size (neurons_output, NEURONS_HIDDEN) with values in range [-1, 1]. WEIGHTS2
# b2: matrix of size (neurons_output, 1) of zeros. BIAS2
def initializeParameters(neurons_input, neurons_output):
    
    w1 = np.random.randn(NEURONS_HIDDEN, neurons_input)*0.01
    b1 = np.zeros((NEURONS_HIDDEN, 1))
    
    w2 = np.random.randn(neurons_output, NEURONS_HIDDEN)*0.01
    b2 = np.zeros((neurons_output, 1))
    
    return w1, b1, w2, b2



# z1: matrix of size (NEURONS_HIDDEN, neurons_input). WEIGHTS1 * INPUT + BIAS
# a1: matrix of size (NEURONS_HIDDEN, neurons_input). Activation function of z1
# z2: matrix of size (neurons_output, NEURONS_HIDDEN). WEIGHTS2 * a1 + BIAS2
# a2: matrix of size (neurons_output, NEURONS_HIDDEN). Activation function of z2
def forward(x, w1, b1, w2, b2):
    
    z1 = np.dot(w1, x) + b1
    a1 = tanh(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    
    return z1, a1, z2, a2



# dw1: matrix of size (NEURONS_HIDDEN, neurons_input).
# db1: matrix of size (NEURONS_HIDDEN, 1).
# dw2: matrix of size (neurons_output, NEURONS_HIDDEN).
# db2: matrix of size (neurons_output, 1).
def backward(x, y, w1, b1, w2, b2, z1, a1, z2, a2):
    
    x_size = x.shape[1]
    
    dz2 = (a2 - y)
    dw2 = (1/x_size)*np.dot(dz2, a1.T)
    db2 = (1/x_size)*np.sum(dz2, axis = 1, keepdims = True)
    
    dz1 = (1/x_size)*np.dot(w2.T, dz2)*tanhDeriv(a1)
    dw1 = (1/x_size)*np.dot(dz1, x.T)
    db1 = (1/x_size)*np.sum(dz1, axis = 1, keepdims = True)

    return dw1, db1, dw2, db2



# w1: matrix of size (NEURONS_HIDDEN, neurons_input) with values in range [-1, 1]. WEIGHTS1
# b1: matrix of size (NEURONS_HIDDEN, 1) of zeros. BIAS1
# w2: matrix of size (neurons_output, NEURONS_HIDDEN) with values in range [-1, 1]. WEIGHTS2
# b2: matrix of size (neurons_output, 1) of zeros. BIAS2
def updateParameters(w1, b1, w2, b2, dw1, db1, dw2, db2):
    
    w1 = w1 - LEARNING_RATE*dw1
    b1 = b1 - LEARNING_RATE*db1
    w2 = w2 - LEARNING_RATE*dw2
    b2 = b2 - LEARNING_RATE*db2
    
    return w1, b1, w2, b2
    

    
# Returns the accuracy of the nn (float).
def accuracy(x, y, w1, b1, w2, b2):
    
    z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

    finalOputputs = np.argmax(a2, 0) 
    
    trueOutputs = np.argmax(y, 0)
    
    a = np.mean(finalOputputs ==  trueOutputs)*100
    
    return a
 


# Returns the accuracy of each different class.
def accuracyPerClass(x, y, w1, b1, w2, b2):   

    z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)

    xBin = np.zeros(a2.shape)  
    xBin[np.argmax(a2, axis=0), np.arange(a2.shape[1])] = 1   
    
    elementsPerClass = y.sum(axis=1)
    
    coincidences = np.logical_and(xBin, y)
    coincidences = coincidences.sum(axis=1)
    
    a = np.ones(elementsPerClass.shape)
    a = np.divide(coincidences, elementsPerClass, out=a, where=elementsPerClass!=0)*100
          
    return a
    

def tanh(x):
    return np.tanh(x)

def softmax(x):
    expX = np.exp(x)
    sm = expX/np.sum(expX, axis = 0) 
    return sm

def tanhDeriv(x):
    return (1 - np.power(np.tanh(x), 2))




main("minst.csv")