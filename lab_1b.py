import numpy as np
import matplotlib.pyplot as plt

n = 100
seed = 42

'''3.1.1 Generation of data'''
def new_data_generation(n):
    mA = np.array([1,0.3])
    mB = np.array([0, -0.1])
    sigmaA = 0.2
    sigmaB = 0.3
    classA = np.zeros((2, n))
    classA[0, :] = np.concatenate((np.random.normal(-mA[0], sigmaA, n//2), np.random.normal(mA[0], sigmaA, n//2)))
    classA[1, :] = np.random.normal(mA[1], sigmaA, n)
    classB = np.zeros((2, n))
    classB[0, :] = np.random.normal(mB[0], sigmaB, n)
    classB[1, :] = np.random.normal(mB[1], sigmaB, n)
    T = np.concatenate((np.ones(n), -np.ones(n)))
    data = np.concatenate((classA, classB), axis=1)
    data, T = shuffle_data_and_target(data, T)
    return shuffle_data_and_target(data, T)

def shuffle_data_and_target(data, target):
    shuffle_indices = np.random.permutation(data.shape[1])
    return data[:,shuffle_indices], target[shuffle_indices]

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
    h_in = W@X
    H = add_bias(activation_function(h_in))
    o_in = V@H
    O = activation_function(o_in)
    
    return H, O


def backward(T, O, H, V, Nhidden):    
    delta_o = (O-T)* (derivative_activation_function(O))
    delta_h = np.dot(V,delta_o)*(derivative_activation_function(H))
    delta_h = delta_h[0:Nhidden]
    return delta_o,delta_h


#initialize weights matrix with number of hidden neurons in the layer editable
def initialize_weights(n_hidden_neurons):
    W = np.random.uniform(-0.5, 0.5, (n_hidden_neurons, 3))
    V = np.random.uniform(-0.5, 0.5, (1, n_hidden_neurons+1))
    return W, V


def weights_update(delta_o, delta_h, X, H, W, V, eta, dw, dv, alpha):
    X = add_bias(X)
    dw = (dw * alpha) - (eta * np.dot(delta_h, X.T))
    #dv = (dv * alpha) - (eta * np.dot(delta_o, H.T))
    W += dw
    V += dv
    return W, V, dw, dv



##do a function to train for an n number of epochs and return a list of weights and error for each epoch
def train(X, T, W, V, eta, epochs,alpha, Nhidden):
    list_error = []
    list_missclass=[]
    list_W=[W.copy()]
    dv, dw = np.ones(V.shape), np.ones(W.shape)
    for i in range(epochs):
        O, H = forward(X, W, V)
        error = 0.5 * np.sum((O - T) ** 2)
        list_error.append(error)
        list_missclass.append(np.sum(np.sign(O)!=T))
        delta_o, delta_h = backward(T, O, H, V, Nhidden)
        W,V, dw, dv = weights_update(delta_o, delta_h, X, H, W, V, eta, dw, dv, alpha)
        list_W.append(W.copy())
    return W,V,list_error,list_W,list_missclass



def all_decision_boundary_plot(X, T, W_list):
    plt.scatter(X[0,:], X[1,:], c=T, cmap=plt.cm.Paired)
    x = np.linspace(-2, 2, 1000)
    for i, W in enumerate(W_list):
        for W_j in W:
            w1, w2, bias = W_j
            y = -(w1*x+bias)/w2
            if i == 0 or i%1000 == 0:
                if i == len(W_list)-1: 
                    plt.plot(x, y, 'r', label = 'Epoch '+str(i))
                else: 
                    plt.plot(x, y, 'k--')
    
##given the two final weights matrices, plot the decision boundaries

def gaussian_data():
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    xx, yy = np.meshgrid(x, y)
    z = np.exp(-(xx**2 + yy**2)/10)-0.5

    fig = plt.figure(figsize=(10,5))
    fig3d = fig.add_subplot(1,2,1, projection='3d')
    fig3d.plot_surface(xx, yy, z, cmap='plasma', shade=True)
    fig3d.set_xlabel('x')
    fig3d.set_ylabel('y')
    fig3d.set_zlabel('z')
    plt.show()

    num = len(x)*len(y)
    T = z.reshape(1, num)
    X = np.concatenate((xx.reshape(1, num), yy.reshape(1, num)), axis=0)

    print(X.shape)
    print(X)
    return X, T

if __name__ == "__main__":
    classA, classB = classes_generation()
    X, T = new_data_generation(100)
    W,V=initialize_weights(3)
    W,V,list_error,list_W,list_missclass=train(X, T, W, V, 0.001, 10000,0.9, 3)
    plot_data(X, T)
    print(list_error[-1])
    all_decision_boundary_plot(X, T, list_W)
    print(list_missclass)
    plt.show()