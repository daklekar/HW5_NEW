#!/usr/bin/env python
# coding: utf-8

# In[645]:


import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from matplotlib import pyplot
import math
import numpy as np
from sklearn import preprocessing 


# In[646]:


DS = pd.read_csv("C:\\Users\\dhruv\\Downloads\\Housing.csv")


# In[647]:


train=DS.sample(frac=0.8,random_state=0) 
test=DS.drop(train.index)


# In[648]:


DS


# In[649]:


train = train[['price','area','bedrooms','bathrooms','stories','parking']]
test = test[['price','area','bedrooms','bathrooms','stories','parking']]


# In[650]:


scaler = preprocessing.MinMaxScaler()


# In[651]:


X1_t = np.array(train.area)
X2_t = np.array(train.bedrooms)
X3_t = np.array(train.bathrooms)
X4_t = np.array(train.stories)
X5_t = np.array(train.parking)

X0_t= np.ones(436)

X = np.vstack([X0_t,X1_t,X2_t,X3_t,X4_t,X5_t])
X = X.T
X = np.array(X)   
x = scaler.fit_transform(X)
X= x
X


# In[652]:


Y_t = np.array(train.price)
Y = Y_t
Y = Y_t.reshape(436,1)
y = scaler.fit_transform(Y)
Y=y


# In[653]:


#weights and bias column
theta = np.array([0., 0., 0., 0., 0., 0.])
theta = theta.reshape(6,1)
theta


# In[654]:


#Data Array
t_c = Y #PRICE(Actual Y's or Data(price))
t_u = X #Epochs(Actual X's ...multidimentional)


# In[655]:


t_c = torch.tensor(t_c, dtype=torch.float64)#.to_numpy()
t_u = torch.tensor(t_u, dtype=torch.float64)


# In[656]:


t_c= np.array(t_c)
t_u= np.array(t_u)
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[657]:


#Weights and Bias as column
b = torch.zeros(1)
w5 = torch.ones(1)
w4 = torch.ones(1)
w3 = torch.ones(1)
w2 = torch.ones(1)
w1 = torch.ones(1)


# In[658]:


theta = [b, w1,w2, w3, w4, w5]
theta = torch.tensor(theta).reshape(6,1)


# In[659]:


#Model definition linear
def model(X, theta):
    return np.matmul(X, theta)


# In[660]:


#Loss function
def loss_func(t_p, t_c):
    var = (t_p - t_c)**2
    return var.mean()


# In[661]:


t_p = model(X, theta)
#t_p = torch.tensor(t_p)


# In[662]:


loss = loss_func(t_p, t_c)
loss


# In[663]:


def b_del(delta):
    b_del = [delta, 0,0,0,0,0]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del
def w5_del(delta):
    b_del = [0, delta,0,0,0,0]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del
def w4_del(delta):
    b_del = [0, 0,delta,0,0,0]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del
def w3_del(delta):
    b_del = [0, 0,0,delta,0,0]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del
def w2_del(delta):
    b_del = [0, 0,0,0,delta,0]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del
def w1_del(delta):
    b_del = [0, 0,0,0,0,delta]
    b_del = torch.tensor(b_del).reshape(6,1)
    return b_del


# In[664]:


delta = 0.7

loss_rate_of_change_b = (loss_func(model(t_u, theta+b_del(delta)), t_c) - 
loss_func(model(t_u, theta-b_del(delta)), t_c))/(2.0 * delta)

loss_rate_of_change_w1 = (loss_func(model(t_u, theta+w1_del(delta)), t_c) - 
loss_func(model(t_u, theta-w1_del(delta)), t_c))/(2.0 * delta)

loss_rate_of_change_w2 = (loss_func(model(t_u, theta+w2_del(delta)), t_c) - 
loss_func(model(t_u, theta-w2_del(delta)), t_c))/(2.0 * delta)

loss_rate_of_change_w3 = (loss_func(model(t_u, theta+w3_del(delta)), t_c) - 
loss_func(model(t_u, theta-w3_del(delta)), t_c))/(2.0 * delta)

loss_rate_of_change_w4 = (loss_func(model(t_u, theta+w4_del(delta)), t_c) - 
loss_func(model(t_u, theta-w4_del(delta)), t_c))/(2.0 * delta)

loss_rate_of_change_w5 = (loss_func(model(t_u, theta+w5_del(delta)), t_c) - 
loss_func(model(t_u, theta-w5_del(delta)), t_c))/(2.0 * delta)


# In[665]:


learning_rate = 1e-2
w1 = w1 - learning_rate * loss_rate_of_change_w1
w2 = w2 - learning_rate * loss_rate_of_change_w2
w3 = w3 - learning_rate * loss_rate_of_change_w3
w4 = w4 - learning_rate * loss_rate_of_change_w4
w5 = w5 - learning_rate * loss_rate_of_change_w5
b = b - learning_rate * loss_rate_of_change_b


# In[666]:


#Partial deriv of loss function 
def dloss_fn(t_p, t_c):
    dsq_diffs=2* (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[667]:


#dm/dw5
def dmodel_dw5(t_u, w5, w4, w3, w2, w1, b):
    return t_u


# In[668]:


#dm/dw4
def dmodel_dw4(t_u, w5, w4, w3, w2, w1, b):
    return t_u


# In[669]:


#dm/dw3
def dmodel_dw3(t_u, w5, w4, w3, w2, w1, b):
    return t_u


# In[670]:


#dm/dw2
def dmodel_dw2(t_u, w5, w4, w3, w2, w1, b):
    return t_u


# In[671]:


#dm/dw1
def dmodel_dw1(t_u, w5, w4, w3, w2, w1, b):
    return t_u


# In[672]:


#dm/db
def dmodel_db(t_u, w5, w4, w3, w2, w1, b):
    return 1.0


# In[673]:


def grad_fn(t_u, t_c, t_p, w5, w4, w3, w2, w1, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw5 = dloss_dtp * dmodel_dw2(t_u, w5, w4, w3, w2, w1, b)
    dloss_dw4 = dloss_dtp * dmodel_dw1(t_u, w5, w4, w3, w2, w1, b)
    dloss_dw3 = dloss_dtp * dmodel_db(t_u, w5, w4, w3, w2, w1, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w5, w4, w3, w2, w1, b)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t_u, w5, w4, w3, w2, w1, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w5, w4, w3, w2, w1, b)
    return torch.stack([dloss_dw5.sum(), dloss_dw4.sum(), dloss_dw3.sum(), dloss_dw2.sum(), dloss_dw1.sum(), dloss_db.sum()])


# In[674]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w5, w4, w3, w2, w1, b = params
        theta = [b, w1,w2, w3, w4, w5]
        theta = torch.tensor(theta).reshape(6,1)
        t_p = model(t_u, theta)
        loss = loss_func(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w5, w4, w3, w2, w1, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print('Params: %f', params)
        print("Grad: %f", grad)
        print('-------------------------------------------------')
    return params


# In[715]:


training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1,1,1,0.5, 0.1, 0.5]),
t_u = t_u,
t_c = t_c)


# In[716]:


X1_t = np.array(test.area)
X2_t = np.array(test.bedrooms)
X3_t = np.array(test.bathrooms)
X4_t = np.array(test.stories)
X5_t = np.array(test.parking)
X0_t= np.ones(109)
X = np.vstack([X0_t,X1_t,X2_t,X3_t,X4_t,X5_t])


# In[717]:


Ex = [0.5937,  0.5937,  0.7912,  0.0937, -0.3063,  0.2912]
Ex = torch.tensor(Ex).reshape(6,1)
X = np.vstack([X0_t,X1_t,X2_t,X3_t,X4_t,X5_t])
X = X.T
X = np.array(X)   
x = scaler.fit_transform(X)
X= x


# In[718]:


Y = np.array(test.price)
Y = Y.reshape(109,1)
y = scaler.fit_transform(Y)
Y = y
# Y.reshape(1,-1)


# In[719]:


p = 108
Wye = np.ones(109)
while(p>0):
    M = X[p,:]  #13300000
    Wye[p] =  np.matmul(M,Ex)
    print('Truth: %f'%Y[p])
    print('Pred: %f'%Wye[p])
    print('--------------------------------------------------------------')
    p=p-1


# In[720]:


pyplot.plot(Y)
pyplot.plot(Wye,'+')


# In[ ]:




