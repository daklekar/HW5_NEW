#!/usr/bin/env python
# coding: utf-8

# In[596]:


import torch
import math
from matplotlib import pyplot as plt
import numpy as np


# In[597]:


#Data Array
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[598]:


#Weights and Bias
w2 = torch.ones(t_u.size(dim=0))
w1 = torch.ones(t_u.size(dim=0))
b = torch.zeros(t_u.size(dim=0))


# In[599]:


#Model definition polynomial
def model(t_u, w2, w1, b):
    return w2*(t_u)**2 + t_u*w1 + b


# In[600]:


#Loss function
def loss_func(t_p, t_c):
    var = (t_p-t_c)**2
    return var.mean()


# In[601]:


t_p = model(t_u, w2, w1, b)
t_p


# In[602]:


loss = loss_func(t_p, t_c)
loss


# In[603]:


delta = 0.1
loss_rate_of_change_b = (loss_func(model(t_u, w2, w1, b+delta), t_c) - loss_func(model(t_u, w2, w1, b-delta), t_c))/(2.0 * delta)
loss_rate_of_change_w1 = (loss_func(model(t_u, w2, w1+delta, b), t_c) - loss_func(model(t_u, w2, w1-delta, b), t_c))/(2.0 * delta)
loss_rate_of_change_w2 = (loss_func(model(t_u, w2+delta, w1 ,b), t_c) - loss_func(model(t_u, w2-delta, w1, b), t_c))/(2.0 * delta)


# In[604]:


print(loss_rate_of_change_b)
print(loss_rate_of_change_w1)
print(loss_rate_of_change_w2)


# In[605]:


#learning_rate = 0.01
w1 = w1 - learning_rate * loss_rate_of_change_w1
w2 = w2 - learning_rate * loss_rate_of_change_w2
b = b - learning_rate * loss_rate_of_change_b


# In[606]:


print(b)
print(w1)
print(w2)


# In[607]:


#Partial deriv of loss function 
def dloss_fn(t_p, t_c):
    dsq_diffs=2* (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[608]:


#dm/dw2
def dmodel_dw2(t_u, w2, w1, b):
    return t_u**2


# In[609]:


#dm/dw1
def dmodel_dw1(t_u, w2, w1, b):
    return t_u


# In[610]:


#dm/db
def dmodel_db(t_u, w2, w1, b):
    return 1.0


# In[611]:


def grad_fn(t_u, t_c, t_p, w2, w1, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w2, w1, b)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t_u, w2, w1, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w2, w1, b)
    return torch.stack([dloss_dw2.sum(), dloss_dw1.sum(), dloss_db.sum()])


# In[612]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w2, w1, b = params
        t_p = model(t_u, w2, w1, b)
        loss = loss_func(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w2, w1, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print('Params: %f', params)
        print("Grad: %f", grad)
        print('-------------------------------------------------')
    return params


# In[615]:


#In this part I was getting na numbers...I HAD to reduce the learning rate to that level......
#rest of the training for LR=0.1,0.01,0.001,0.0001 is done in the following sub-parts of the HW like Problem 1.1,1.2,etc.

t_un = 0.1 * t_u
training_loop(
n_epochs = 500,
learning_rate = 1e-4,
params = torch.tensor([1, 1, 0]),
t_u = t_un,
t_c = t_c)


# In[614]:


plt.plot(np.sort(t_u.numpy()), t_c.numpy(), '.', color = "navy");
plt.plot(np.sort(t_u.numpy()), np.sort(t_p.detach().numpy()), color = "r")

