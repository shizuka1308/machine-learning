import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from numpy import exp
import time
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    expVal = np.exp(-z)
    sigmoid_z = 1/(1 + expVal)

    return sigmoid_z


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all
     '.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    preprocTrain = np.zeros(shape=(50000, 784))
    preprocValid = np.zeros(shape=(10000, 784))
    preprocTest = np.zeros(shape=(10000, 784))
    preprocLabelTrain = np.zeros(shape=(50000,))
    preprocLabelValid = np.zeros(shape=(10000,))
    preprocLabelTest = np.zeros(shape=(10000,))

    lengthOfTrain = 0
    lengthOfValid = 0
    lengthTest = 0
    lengthTrainLabel = 0
    lengthLabelValied = 0

    for iterator in mat:
        if "train" in iterator:

            preprocTrain[lengthOfTrain:lengthOfTrain +
                         (len(mat.get(iterator))) - 1000] = (mat.get(iterator))[(np.random.permutation(range((mat.get(iterator)).shape[0])))[1000:], :]
            lengthOfTrain = lengthOfTrain + (len(mat.get(iterator))) - 1000

            preprocLabelTrain[lengthTrainLabel:lengthTrainLabel +
                              (len(mat.get(iterator))) - 1000] = iterator[-1]
            lengthTrainLabel = lengthTrainLabel + \
                (len(mat.get(iterator))) - 1000

            preprocValid[lengthOfValid:lengthOfValid +
                         1000] = (mat.get(iterator))[(np.random.permutation(range((mat.get(iterator)).shape[0])))[0:1000], :]
            lengthOfValid = lengthOfValid + 1000

            preprocLabelValid[lengthLabelValied:
                              lengthLabelValied + 1000] = iterator[-1]
            lengthLabelValied = lengthLabelValied + 1000

        if "test" in iterator:
            preprocLabelTest[lengthTest:lengthTest +
                             len(mat.get(iterator))] = iterator[-1]
            preprocTest[lengthTest:lengthTest +
                        len(mat.get(iterator))] = (mat.get(iterator))[np.random.permutation(range((mat.get(iterator)).shape[0]))]
            lengthTest = lengthTest + len(mat.get(iterator))

    permanentTrain = np.random.permutation(range(preprocTrain.shape[0]))
    train_data = np.double(preprocTrain[permanentTrain]) / 255
    train_label = preprocLabelTrain[permanentTrain]

    vpermanentValid = np.random.permutation(range(preprocValid.shape[0]))
    validation_data = np.double(preprocValid[vpermanentValid]) / 255
    validation_label = preprocLabelValid[vpermanentValid]

    tpermanentTest = np.random.permutation(range(preprocTest.shape[0]))
    test_data = np.double(preprocTest[tpermanentTest]) / 255
    test_label = preprocLabelTest[tpermanentTest]

    # Feature selection
    # Your code here.
    dataTotal = np.array(np.vstack((train_data, validation_data, test_data)))
    duples = np.all(dataTotal == dataTotal[0, :], axis=0)
    featureSelection = np.where(duples == False)
    dataTotal = dataTotal[:, ~duples]

    selected_features = np.array([])
    selected_features = featureSelection[0]

    train_data = dataTotal[0:len(train_data), :]
    validation_data = dataTotal[len(train_data): (
        len(train_data) + len(validation_data)), :]
    test_data = dataTotal[(len(train_data) + len(validation_data))
                           : (len(train_data) + len(validation_data) + len(test_data)), :]
    print('preprocess done')

    return selected_features, train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    transw1 = np.transpose(w1)
    transw2 = np.transpose(w2)

    shapeData = train_data.shape[0]
    wholeData = (shapeData, 1)
    bias = np.ones(wholeData, dtype=np.float64)
    dataTrainBias = np.concatenate((train_data, bias), axis=1)
    jazz = sigmoid(np.matmul(dataTrainBias, transw1))

    biashid = np.ones(shape=(jazz.shape[0], 1), dtype=np.float64)
    zjBias = np.concatenate((jazz, biashid), axis=1)

    olay = sigmoid(np.matmul(zjBias, transw2))

    someShape = (train_data.shape[0])
    wholeData = (someShape, 10)
    ylay = np.zeros(shape=(wholeData), dtype=np.float64)

    for i in range(ylay.shape[0]):
        for j in range(ylay.shape[1]):
            if j == training_label[i]:
                ylay[i][j] = 1.0

    glitch = (np.sum(ylay*np.log(olay) + (1-ylay)*np.log(1-olay))) / \
        (-1*(train_data.shape[0]))
    reg_glitch = glitch + ((np.sum(np.square(w1)) + np.sum(np.square(w2)))
                           * lambdaval)/(2*(train_data.shape[0]))

    obj_val = reg_glitch
    obj_grad = np.array([])
    sr = ((1-zjBias[:, 0:n_hidden])*zjBias[:, 0:n_hidden]) * \
        (np.matmul(olay-ylay, w2[:, 0:n_hidden]))

    regularw2Gradient = (np.matmul(np.transpose(olay-ylay),
                                   zjBias) + (lambdaval*w2))/(train_data.shape[0])
    regularw1Gradient = (np.matmul(np.transpose(sr), dataTrainBias) +
                         lambdaval*w1)/(train_data.shape[0])
    obj_grad = np.concatenate(
        (regularw1Gradient.flatten(), regularw2Gradient.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    # Your code here
    shapeData = data.shape[0]
    wholeData = (shapeData, 1)
    labels = np.full(wholeData, 0)
    bias = np.ones(wholeData, dtype=np.float64)
    data_bias = np.concatenate((data, bias), axis=1)

    transw1 = np.transpose(w1)
    transw2 = np.transpose(w2)

    zj_bias = np.concatenate(
        ((sigmoid(np.matmul(data_bias, transw1))), np.ones(shape=((sigmoid(np.matmul(data_bias, transw1))).shape[0], 1), dtype=np.float64)), axis=1)

    olay = sigmoid(np.matmul(zj_bias, transw2))
    n = 0
    while n < ((olay).shape[0]):
        labels[n] = np.argmax(olay[n])
        n = n+1

    return labels


"""**************Neural Network Script Starts here********************************"""

selected_features, train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 10

# n_hidden_array = [4,8,12,16,20]

# for x in n_hidden_array:
#    for y in range(0,70,10):

# set the number of nodes in hidden unit (not including bias unit)
# n_hidden = x # values - 4,8,12,16,20
n_hidden = 20  # Optimal hidden units

# set the regularization hyper-parameter
# lambdaval = y # values - 0 to 60, 10
lambdaval = 10  # Optimal lambda value

# Note current time
t1 = time.time()

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate(
    (initial_w1.flatten(), initial_w2.flatten()), 0)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                     args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1))                 :].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

print("\n Weight vector w1:"+str(w1))
print("\n Weight vector w2:"+str(w2))

# find the accuracy on Training Dataset

print('\n lambda:'+str(lambdaval)+'\n hidden layers:'+str(n_hidden))
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label ==
      train_label.reshape(train_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label ==
      validation_label.reshape(validation_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label ==
      test_label.reshape(test_label.shape[0], 1)).astype(float))) + '%')

t2 = time.time()

print('\n Time taken:'+str(t2-t1))

store_obj = dict([("selected_features", selected_features), ("n_hidden", n_hidden), ("w1", w1),
                  ("w2", w2), ("lambdaval", lambdaval)])
pickle.dump(store_obj, open('params.pickle', 'wb'), protocol=3)
