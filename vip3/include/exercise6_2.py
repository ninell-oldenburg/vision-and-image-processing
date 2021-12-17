#!/usr/bin/env python
# coding: utf-8

# For both all datasets manipulation and reshaping is very important to maintain good
# performances. Both Matlab and Python/Numpy allow you to extract a subarray as a list
# of value, and affects a list of value to an image subarray. Woodham’s estimation can be
# vectorised. You may want to look at integration source code to see how Python allows for
# these type of manipulations.

# Each of these functions are reasonably well documented using docstrings, so, after importing
# ps utils.py, you can invoke help(ps utils) for the entire documentation, or for a specific
# function such as help(ransac 3dvector) etc...

# ### Exercise 6

# 
#     Buddha is real dataset, with exactly 10 images.
# 

# In[1]:


import os
import numpy as np
import ps_utils as utils
import skimage
import matplotlib.pyplot as plt


# In[ ]:


filename = "Buddha.mat"

path = os.path.join("utils", filename)
# Reading shiny vase matlab file
# I - 3D array of image size (m,n,k) where k is views
# mask - with records of intensity data
# S - light vectors 
I, mask, S = utils.read_data_file(path)

m = np.zeros((I.shape[0], I.shape[1], 3))
al = np.zeros((I.shape[0], I.shape[1], 3))
normals = np.zeros((I.shape[0], I.shape[1], 3))

for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        if mask[i,j]!=0:
            model, inliers, best_fit = utils.ransac_3dvector(data=(I[i,j,:], S), threshold = 50.0)
            if np.linalg.norm(model)!=0:
                m[i,j] = model 
                al[i,j] = (np.linalg.norm(model))
                normal = (1 /  np.linalg.norm(model)) * model
                normals[i,j] = normal

                
normal_ones = normals.copy()
al_ones = al.copy()
m_ones = m.copy()


normal_ones[normal_ones==0]=1

n1 = normal_ones[:,:,0]
n2 = normal_ones[:,:,1]
n3 = normal_ones[:,:,2]

z = utils.simchony_integrate(n1, n2, n3, mask)
# n1, n2, n3: nympy float arrays the 3 components of the normal. They must be 2D arrays
# Copying the mask
z = utils.simchony_integrate(n1, n2, n3, mask)
# z_unbiased = utils.unbiased_integrate(n1, n2, n3, mask)


utils.display_surface(z, albedo=None)


#     Try then to make the estimated normal field smoother using the smooth normal field() function. You may experiment with the iters parameter.

# In[ ]:
smoothed_normals = utils.smooth_normal_field(n1, n2, n3, mask)

n1_s , n2_s, n3_s = smoothed_normals

# Solving for depth and displaying the image
z = utils.simchony_integrate(n1_s, n2_s, n3_s, mask)
utils.display_surface(z, albedo=None)

smoothed_normals = utils.smooth_normal_field(n1, n2, n3, mask, iters = 500)

n1_s , n2_s, n3_s = smoothed_normals

# Solving for depth and displaying the image
z = utils.simchony_integrate(n1_s, n2_s, n3_s, mask)
utils.display_surface(z, albedo=None)


smoothed_normals = utils.smooth_normal_field(n1, n2, n3, mask, iters = 1000)

n1_s , n2_s, n3_s = smoothed_normals

# Solving for depth and displaying the image
z = utils.simchony_integrate(n1_s, n2_s, n3_s, mask)
utils.display_surface(z, albedo=None)



