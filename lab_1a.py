import numpy as np
import matplotlib.pyplot as plt

n = 100
seed = 42


'''3.1.1 Generation of linearly-separable data'''

def classes_generation():
    mA = np.array([2, 2])
    mB = np.array([1, 1])
    sigmaA = 0.5
    sigmaB = 0.5
    classA = np.zeros((2, n))
    np.random.seed(seed)
    classA[0, :] = np.random.normal(mA[0], sigmaA, n)
    np.random.seed(seed+1)
    classA[1, :] = np.random.normal(mA[1], sigmaA, n)
    classB = np.zeros((2, n))
    np.random.seed(seed+2)
    classB[0, :] = np.random.normal(mB[0], sigmaB, n)
    np.random.seed(seed+3)
    classB[1, :] = np.random.normal(mB[1], sigmaB, n)

    return classA, classB


def plot_data(X, T):
    ls = plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    plt.legend(handles=ls.legend_elements()[0], labels=['Class A', 'Class B'])
    plt.grid(True)
    plt.title("Generated samples")
    plt.show()


def data_generation(classA, classB):
    X = np.concatenate((classA, classB), axis=1)
    T = np.array([1]*n + [-1]*n)
    
    # shuffle data points
    np.random.seed(seed)
    idx = list(range(2*n))
    np.random.shuffle(idx)
    X = X[:, idx]
    T = T[idx]

    return X, T

def add_bias(X):
    X = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0)
    return X

'''3.1.2 Classification with a single-layer perceptron and analysis'''

def initialize_weights(n_output, n_input):
    np.random.seed(12)
    weights = np.random.normal(size=(n_output,n_input), loc=0, scale=1)
    weights = [np.array([5, 8, 10])]
    return weights


def batch_perceptron_learning(X, T, weights, eta, epochs):
    W = weights.copy()
    X = add_bias(X)
    errors_list = [np.mean(abs(np.sign(W@X)-T)/2)]
    weights_list = [W[0]]
    for epoch in range(epochs):
        delta_W = -eta*(np.sign(W@X)-T)@X.T
        W += delta_W
        errors_list.append(np.mean(abs(np.sign(W@X)-T)/2))
        W_l = W.copy()
        weights_list.append(W_l[0])

    return W, weights_list, errors_list


def online_perceptron_learning(X, T, weights, eta, epochs):
    W = weights.copy()
    X = add_bias(X)
    errors_list = []
    weights_list = [W[0]]
    for epoch in range(epochs):
        var = 0
        for i in range(X.shape[1]):
            if T[i] != np.sign(W@X[:,i]):
                var += 1
            delta_W = -eta*(np.sign(W@X[:,i])-T[i])*X[:,i].T
            W += delta_W
        W_l = W.copy()
        weights_list.append(W_l[0])
        errors_list.append(var)
    return W, weights_list, errors_list


def online_delta_rule(X, T, weights, eta, epochs):
    W = weights.copy()
    X = add_bias(X)
    errors_list = [np.mean((W@X-T)**2/2)]
    weights_list = [W[0]]
    for epoch in range(epochs):
        #er=0
        for i in range(X.shape[1]):
            delta_W = -eta*(W@X[:,i]-T[i])*X[:,i].T
            W += delta_W
            #er+=abs((W@X[:,i])-T[i])
        #errors_list.append(er/X.shape[1])
        errors_list.append(np.mean((W@X-T)**2/2))
        W_l = W.copy()

        weights_list.append(W_l[0])
    return W, weights_list, errors_list


def batch_delta_rule(X, T, weights, eta, epochs):
    W = weights.copy()
    X = add_bias(X)
    print(X.shape, W[0].shape)
    errors_list = [np.mean((W@X-T)**2/2)]
    weights_list = [W[0]]
    for epoch in range(epochs):
        delta_W = 0
        delta_W = -eta*(W@X-T)@X.T
        W += delta_W
        errors_list.append(np.mean((W@X-T)**2/2))
        W_l = W.copy()
        weights_list.append(W_l[0])

    return W, weights_list, errors_list


def plot_errors(errors_list):
    print("Error epoch 50: ", errors_list[49])
    #plt.text(0.95, 0.95, "Error epoch 50: "+str(round(errors_list[49], 3)), fontsize=12, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.plot(errors_list)
    plt.title("Loss function")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

def decision_boundary_plot(X, T, W_list):
    plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    x = np.linspace(min(X[0,:]), max(X[0,:]), 100)
    w1, w2, bias = W_list[-1]
    y = -(w1*x+bias)/w2
    plt.plot(x, y, 'k')
    plt.title("Final decision boundary")
    plt.show()

def all_decision_boundary_plot(X, T, W_list):
    plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    x = np.linspace(min(X[0,:]), max(X[0,:]), 100)
    for i, W in enumerate(W_list):
        w1, w2, bias = W
        y = -(w1*x+bias)/w2
        if i == 0 or i%50 == 0:
            if i == len(W_list)-1: plt.plot(x, y, 'r', label = 'Epoch '+str(i))
            else: plt.plot(x, y, 'k--')
    plt.title("Decision boundaries by epoch for delta rule with eta=0.01")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

if __name__ == "__main__":
    classA, classB = classes_generation()
    X, T = data_generation(classA, classB)
    plot_data(X, T)
    weights = initialize_weights(1, X.shape[0]+1)
    #W, W_list, err_list = batch_perceptron_learning(X, T, weights, 0.01, 100)
    #W, W_list, err_list = online_perceptron_learning(X, T, weights, 0.01, 100)
    W, W_list, err_list = online_delta_rule(X, T, weights, 0.0005, 200)
    #W, W_list, err_list = batch_delta_rule(X, T, weights, 0.0005, 200)
    all_decision_boundary_plot(X, T, W_list)
    plot_errors(err_list)