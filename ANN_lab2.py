import numpy as np
import matplotlib.pyplot as plt

n = 100
seed = 42

'''3.1.1 Generation of data'''
def classes_generation():
    mA = np.array([0.8,0.2])
    mB = np.array([0, -0.1])
    sigmaA = 0.2
    sigmaB = 0.3
    classA = np.zeros((2, n))
    np.random.seed(seed)
    classA[0, :] = np.concatenate((np.random.normal(-mA[0], sigmaA, n//2), np.random.normal(mA[0], sigmaA, n//2)))
    classA[1, :] = np.random.normal(mA[1], sigmaA, n)
    #remove 25% of the points
    # classA = np.delete(classA, np.random.choice(classA.shape[1], int(n/2), replace=False), axis=1)
    # for i in range(n):
    #     if classA[0,i]>0:
    #         index1.append(i)
    #     else:
    #         index2.append(i)
    #generate the 20% of the index in index1 and the 80% of index in index2 and remove them from the classA
    # np.random.shuffle(index1)
    # np.random.shuffle(index2)
    # classA = np.delete(classA, index1[:int(0.2*len(index1))], axis=1)
    # classA = np.delete(classA, index2[:int(0.8*len(index2))], axis=1)
    np.random.seed(seed+1)
    classB = np.zeros((2, n))
    np.random.seed(seed+2)
    classB[0, :] = np.random.normal(mB[0], sigmaB, n)
    np.random.seed(seed+3)
    classB[1, :] = np.random.normal(mB[1], sigmaB, n)
    # #remove 25% of the points
    # classB = np.delete(classB, np.random.choice(classB.shape[1], int(n/2), replace=False), axis=1)
    return classA, classB

def plot_data(X, T):
    ls = plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    plt.legend(handles=ls.legend_elements()[0], labels=['Class A', 'Class B'])
    plt.grid(True)
    plt.title("Generated samples")
    plt.show()


def data_generation(classA, classB):
    X = np.concatenate((classA, classB), axis=1)
    T = np.array([1]*int(n) + [-1]*int(n))
    # shuffle data points
    np.random.seed(seed)
    idx = list(range(int(2*n)))
    np.random.shuffle(idx)
    X = X[:, idx]
    T = T[idx]

    return X, T


def activation_function(x):
    return 2/(1+np.exp(-x))-1
def derivative_activation_function(x):
    return 0.5*(1+activation_function(x))*(1-activation_function(x))

def add_bias(X):
    return np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

def forward(X, W, V):
    X = add_bias(X)
    Z = activation_function(np.dot(W, X))
    Z = add_bias(Z)
    Y = activation_function(np.dot(V, Z))
    return Y, Z


def backward(X, T, Y, Z, V):
    X = add_bias(X)
    delta = (Y - T) * derivative_activation_function(Y)
    deltaV = np.dot(delta, Z.T)
    delta = np.dot(V.T, delta) * derivative_activation_function(Z)
    delta = np.delete(delta, 0, 0)
    deltaW = np.dot(delta, X.T)
    return deltaW, deltaV


#initialize weights matrix with number of hidden neurons in the layer editable
def initialize_weights(n_hidden_neurons):
    W = np.random.uniform(-0.5, 0.5, (n_hidden_neurons, 3))
    V = np.random.uniform(-0.5, 0.5, (1, n_hidden_neurons+1))
    return W, V


##do a function to train for an n number of epochs and return a list of weights and error for each epoch
def train(X, T, W, V, eta, epochs):
    list_error = []
    list_W=[W.copy()]
    list_V=[V.copy()]
    for i in range(epochs):
        Y, Z = forward(X, W, V)
        error = 0.5 * np.sum((Y - T) ** 2)
        list_error.append(error)
        deltaW, deltaV = backward(X, T, Y, Z,V)
        W = W - eta * deltaW
        V = V - eta * deltaV
        list_W.append(W.copy())
        list_V.append(V.copy())
    return W,V,list_error,list_W,list_V

def decision_boundary_plot(X, T, W_list):
    plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    x = np.linspace(min(X[0,:]), max(X[0,:]), 100)
    w1, w2, bias = W_list[-1]
    y = -(w1*x+bias)/w2
    plt.plot(x, y, 'k')
    plt.title("Final decision boundary")
    plt.show()

##given the two final weights matrices, plot the decision boundaries

if __name__ == "__main__":
    classA, classB = classes_generation()
    X, T = data_generation(classA, classB)
    W,V=initialize_weights(2)
    W,V,list_error,list_W,list_V=train(X, T, W, V, 0.01, 1000)
    decision_boundary_plot(X, T, list_V[-1])
    plot_data(X, T)
    
    
    
   