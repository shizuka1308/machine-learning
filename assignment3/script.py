import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

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
    """

    # loads the MAT object as a Dictionary
    mat = loadmat('/content/sample_data/mnist_all.mat')

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation,
                        :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) *
                         n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation,
                   :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * \
            np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    ram = np.empty((n_data, 1))
    ram[:] = 1
    X = np.append(ram, train_data, axis=1)
    weight = initialWeights.reshape(n_features + 1, 1)
    theta_value = sigmoid(np.matmul(X, weight))
    function_error = labeli * \
        np.log(theta_value) + (1 - labeli) * np.log(1.0 - theta_value)
    sum_value = - 1 * np.sum(function_error)
    error = (sum_value) / n_data
    theta_labeli_val = (theta_value - labeli)
    theta_sum_val = np.sum(theta_labeli_val*X, axis=0)
    error_grad = theta_sum_val / n_data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    ram = np.empty((data.shape[0], 1))
    ram[:] = 1
    Xis = np.append(ram, data, axis=1)

    probability = sigmoid(np.matmul(Xis, W))
    loop = np.argmax(probability, axis=1)
    label = loop.reshape((data.shape[0], 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_class = 10
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    ram = np.empty((n_data, 1))
    ram[:] = 1
    Xis = np.append(ram, train_data, axis=1)

    Wis = params.reshape(n_feature + 1, n_class)

    matmul_val = np.matmul(Xis, Wis)
    up_val = np.exp(matmul_val)

    low_val = np.sum(up_val, axis=1)
    low_val = low_val.reshape(low_val.shape[0], 1)

    theta_value = up_val/low_val

    sum_inner = np.sum(Y*np.log(theta_value))

    error = -1 * (np.sum(sum_inner))
    theta_lablei_val = (theta_value - labeli)
    error_grad = np.matmul(Xis.T, theta_lablei_val)
    error_grad = error_grad.ravel()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    row = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    ram = np.empty((row, 1))
    ram[:] = 1
    Xis = np.append(ram, data, axis=1)
    matmul_val = np.matmul(Xis, W)
    exp_val = np.exp(matmul_val)
    tis = np.sum(exp_val, axis=1)
    tis = tis.reshape(tis.shape[0], 1)
    matmul_value = np.matmul(Xis, W)
    exp_val = np.exp(matmul_value)
    value_theta = exp_val/tis

    label = np.argmax(value_theta, axis=1)
    label = label.reshape(row, 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    print('Class', i+1)
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights,
                         jac=True, args=args, method='CG', options=opts)
    print(nn_params['fun'])
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' +
      str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 *
      np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' +
      str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

# for testing data
n_test = test_data.shape[0]

n_feature_test = test_data.shape[1]

Y_test = np.zeros((n_test, n_class))
for i in range(n_class):
    Y_test[:, i] = (test_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature_test + 1, n_class))
initialWeights = np.zeros((n_feature_test + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    print('Class', i+1)
    labeli = Y_test[:, i].reshape(n_test, 1)
    args = (test_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights,
                         jac=True, args=args, method='CG', options=opts)
    print(nn_params['fun'])
    W[:, i] = nn_params.x.reshape((n_feature_test + 1,))

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

index = np.random.randint(50000, size=10000)
data_training_svm = train_data[index, :]
label_training_svm = train_label[index, :]

model_lin = svm.SVC(kernel='linear')
model_lin.fit(data_training_svm, label_training_svm)

print('\n**********Linear Kernel**********\n')
model_lin_val_train = model_lin.score(train_data, train_label)
train_acc = str(100 * model_lin_val_train)
model_lin_val_valid = model_lin.score(validation_data, validation_label)
valid_acc = str(100 * model_lin_val_valid)
model_lin_val_test = model_lin.score(test_data, test_label)
test_acc = str(100 * model_lin_val_test)

print('\n Training Accuracy ***>>>' + train_acc + '%')
print('\n Validation Accuracy ***>>>' + valid_acc + '%')
print('\n Testing Accuracy ***>>>' + test_acc + ' %')

mod_value_rbf = svm.SVC(kernel='rbf', gamma=1.0)
mod_value_rbf.fit(data_training_svm, label_training_svm)

print('\n**********RBF Kernel Gamma Value = 1 **********\n')
mod_val_rbf_train = mod_value_rbf.score(data_training_svm, label_training_svm)
mod_val_rbf_valid = mod_value_rbf.score(validation_data, validation_label)
mod_val_rbf_test = mod_value_rbf.score(test_data, test_label)

print('\n Training Accuracy ***>>>' + str(100 * mod_val_rbf_train) + '%')
print('\n Validation Accuracy ***>>>' + str(100 * mod_val_rbf_valid) + '%')
print('\n Testing Accuracy ***>>' + str(100 * mod_val_rbf_test) + '%')

mod_value_rbf1 = svm.SVC(kernel='rbf', gamma='auto')
mod_value_rbf1.fit(data_training_svm, label_training_svm)

mod_val_rbf1_train = mod_value_rbf1.score(train_data, train_label)
mod_val_rbf1_valid = mod_value_rbf1.score(validation_data, validation_label)
mod_val_rbf1_test = mod_value_rbf1.score(test_data, test_label)
print('\n**********RBF Kernel Gamma Value = default **********\n')
print('\n Training Accuracy ***>>>' + str(100 * mod_val_rbf1_train) + '%')
print('\n Validation Accuracy ***>>' + str(100 * mod_val_rbf1_valid) + '%')
print('\n Testing Accuracy ***>>' + str(100 * mod_val_rbf1_test) + '%')

acc = np.zeros((11, 3), float)
C_val = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
input_value = 0

for cing in C_val:
    print("C Value: \n", cing)
    mod_value_rbf2 = svm.SVC(kernel='rbf', C=cing)
    mod_value_rbf2.fit(data_training_svm, label_training_svm.ravel())
    if input_value <= 10:
        mod_val_rbf2_train = mod_value_rbf2.score(train_data, train_label)
        acc[input_value][0] = 100 * mod_val_rbf2_train
        mod_val_rbf2_valid = mod_value_rbf2.score(
            validation_data, validation_label)
        acc[input_value][1] = 100 * mod_val_rbf2_valid
        mod_val_rbf2_test = mod_value_rbf2.score(test_data, test_label)
        acc[input_value][2] = 100 * mod_val_rbf2_test

        print('\n**********RBF Kernel Gamma Value = default and C = ' +
              str(cing) + '**********\n')
        print('\n Training Accuracy -->' + str(acc[input_value][0]) + '%')
        print('\n Validation Accuracy -->' + str(acc[input_value][1]) + '%')
        print('\n Testing Accuracy -->' + str(acc[input_value][2]) + '%')

    input_value = input_value + 1

'''
Figure and Title   
'''
plt.figure(figsize=(16, 12))
plt.title('Accuracy vs C', pad=10, fontsize=20, fontweight='bold')

plt.xlabel('Value of C', labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy',   labelpad=20, weight='bold', size=15)

plt.xticks(np.array([1,  10,  20,  30,  40,  50,
           60,  70,  80,  90, 100]), fontsize=15)
plt.yticks(np.arange(85, 100, step=0.5),  fontsize=15)


plt.plot(C_val, acc[:, 0], color='m')
plt.plot(C_val, acc[:, 1], color='y')
plt.plot(C_val, acc[:, 2], color='k')

plt.legend(['Training_Data', 'Validation_Data', 'Test_Data'])

mod_value_rbfel_full = svm.SVC(kernel='rbf', gamma='auto', C=70)
mod_value_rbfel_full.fit(train_data, train_label.ravel())

print('**********-\n RBF with FULL training set with best C : \n**********---')
mod_value_rbfel_full_train = mod_value_rbfel_full.score(
    train_data, train_label)
mod_value_rbfel_full_valid = mod_value_rbfel_full.score(
    validation_data, validation_label)
mod_value_rbfel_full_test = mod_value_rbfel_full.score(test_data, test_label)
print('\n Training Accuracy ****>>>' +
      str(100 * mod_value_rbfel_full_train) + '%')
print('\n Validation Accuracy ****>>>' +
      str(100 * mod_value_rbfel_full_valid) + '%')
print('\n Testing Accuracy ****>>>' +
      str(100 * mod_value_rbfel_full_test) + '%')


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b,
                     jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 *
      np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 *
      np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 *
      np.mean((predicted_label_b == test_label).astype(float))) + '%')
