import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    modifiedArray = [tuple(row) for row in y]
    uniques = np.unique(modifiedArray)
    k = uniques.size
    d = len(X[0])
    cv = np.zeros((d,d))
    m = np.zeros((d, k))
    iterator = 1
    while iterator < k+1:
        ci = np.where(y==iterator)[0]
        cd = X[ci,:]
        m[:, iterator-1] = np.average(cd, axis=0).T
        iterator += 1

    cv = np.cov(X.transpose())

    return m,cv

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    modifiedArray = [tuple(row) for row in y]
    uniques = np.unique(modifiedArray)
    k = uniques.size
    d = len(X[0])
    m = np.zeros((d, k))
    cv = []

    iterator = 1
    while iterator < k+1:
        ci = np.where(y==iterator)[0]
        cd = X[ci,:]
        m[:, iterator-1] = np.average(cd, axis=0).T
        modified = cd.T
        cov = np.cov(modified)
        cv += [cov]
        iterator += 1

    return m,cv

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N = np.shape(Xtest)[0]
    #N = len(Xtest[0])
    cc = np.shape(means)[1]
    #cc = len(Xtest[1])
    ypred = np.zeros((N, 1))
    notCorrect = 0.0
    invCov = np.linalg.inv(covmat)
    ytest = ytest.astype(int)

    iterator = 1
    #jiterator = 1
    while iterator < N+1:
        pdf = 0
        predClass = 0
        test_X = (Xtest[iterator-1,:]).T
        for jiterator in range (1, cc+1):
            value1 = (test_X - means[:, jiterator-1])
            value2 = test_X - means[:, jiterator-1]
            dotVal = dotVal = np.matmul(value2.T,invCov)
            dotVal2 = np.matmul(dotVal, value1)
            ans = np.exp((-1/2) * dotVal2)
            if (ans > pdf):
                pdf = ans
                ypred[iterator-1,:] = jiterator
                predClass = jiterator
                                
        if (predClass != ytest[iterator-1]):
            notCorrect = notCorrect + 1

        iterator += 1
    diff = (N - notCorrect)
    accu = diff/N


    return accu * 100, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #classCount = np.shape(means)[1]
    cc = len(means[1])
    N = np.shape(Xtest)[0]

    notPermanent = np.zeros((N,cc))
    prediClass = np.zeros((N,1))

    not_correct = 0.0
    covmat_arr = np.array(covmats)
    
    iterator = 0
    while iterator < cc:
        for jiterator in range(0,N):
            transposeData1 = (Xtest[jiterator,:]-(means[:,iterator]).T)
            linearData = np.linalg.inv(covmat_arr[iterator])
            transposeData2 = (means[:,iterator]).T
            dotVal1 = np.matmul(Xtest[jiterator,:]-transposeData2, linearData)
            p = np.matmul(dotVal1, transposeData1)
            detData = np.linalg.det(covmat_arr[iterator])
            sqrtData = detData**.5
            notPermanent[jiterator,iterator] = 1/(2*pi)* sqrtData * np.exp(p)

        iterator = iterator + 1


    asforData = np.asfortranarray(notPermanent)
    prediClass = (asforData.argmin(axis=1)) + 1

    iteration = 0
    while iteration < N:
        if(ytest[iteration] != prediClass[iteration]):
            not_correct = not_correct + 1
        iteration = iteration + 1
    
    ypred = prediClass.reshape(Xtest.shape[0],1) 

    diff = (N - not_correct)
    accu = diff/N

    return accu * 100, ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    pinvData = np.linalg.pinv(X)
    w = np.matmul(pinvData, y)                                                  
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD   
    identityData =  np.identity(X.shape[1])
    dotData = np.matmul(X.T, X)
    w = np.linalg.inv(dotData + (lambd * identityData))
    dotVal1 = np.matmul(w, X.T)
    w = np.matmul(dotVal1, y)                                               
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    newy = np.matmul(Xtest, w)
    error = (ytest - newy) * (ytest - newy) 
    mse = np.sum(error, axis = 0) / Xtest.shape[0]
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD  
    w = w.reshape(65,1)
    et = np.matmul(X,w)
    XtX = np.matmul(X.T, X)

    dotVal1 = np.matmul((et - y).T, (et - y))
    dotVal2 = np.matmul(w.T, w)
    error = 0.5 * dotVal1 + 0.5 * lambd * dotVal2

    matmulVal1 = np.matmul(XtX, w)
    matmulVal2 = np.matmul(X.T, y)
    gradError = matmulVal1 - matmulVal2 + lambd * w
    gradError = gradError.ravel()
  
    return error, gradError

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.ones((N, p+1))
    #Xp = np.full((N, p+1),1)
    iterator = 1
    while iterator < p+1:
        Xp[:, iterator] = x**iterator
        iterator = iterator + 1
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
train_mle_without_intercepts = testOLERegression(w,X,y)
train_mle_with_intercepts = testOLERegression(w_i,X_i,y)
print('Linear Regression MSE without intercept for train data'+str(train_mle_without_intercepts))
print('Linear Regression MSE with intercept for train data'+str(train_mle_with_intercepts))
print('Linear Regression MSE without intercept for test data'+str(mle))
print('Linear Regression MSE with intercept for test data '+str(mle_i))




# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
train_size,test_size = sys.maxsize,sys.maxsize
train_lambda_value,test_lambda_value = 0,0

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if (mses3_train[i] < train_size):
        train_size = mses3_train[i]
        train_lambda_value = lambd
    elif (mses3[i] < test_size):
        test_size = mses3[i]
        test_lambda_value = lambd
    i = i + 1

fig = plt.figure(figsize=[12,6])
train_data_3 = train_size
train_lambda = train_lambda_value
test_data_3 = test_size
test_lambda = test_lambda_value
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()
print('Ridge Regression MSE for train data:', train_data_3)
print("lambda value", train_lambda)
print('Ridge Regression MSE for test data :', test_data_3)
print("lambda value", test_lambda)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
train_data_4 = np.amin(mses4_train)
test_data_4 = np.amin(mses4)
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
print("Ridge Regression Gradient Descent MSE train :" , train_data_4)
print("Ridge Regression Gradient Descent MSE test :" ,test_data_4 )

# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
train_data_5 = np.amin(mses5)
test_data_5 = np.amin(mses5_train)
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
print("Non Linear MSE for train data",train_data_5)
print("Non Linear MSE for test data",test_data_5)
