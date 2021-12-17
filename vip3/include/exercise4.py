#!/usr/bin/env python
# coding: utf-8

# ### Exercise 4

# shiny vase is a synthetic and clean dataset

# In[2]:


#!pip install mayavi
#!jupyter nbextension install --py mayavi --user
#!jupyter nbextension enable --py mayavi --user


# In[3]:


import ps_utils as utils
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


utils


# In[9]:


filename = "shiny_vase.mat"
path = os.path.join("utils", filename)

# Reading shiny vase matlab file
# I - 3D array of image size (m,n,k) where k is views
# mask - with records of intensity data
# S - light vectors 
I, mask, S = utils.read_data_file(path)


# In[10]:


# show the 3 available images of vase
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()

# images = data.stereo_motorcycle()
ax[0].imshow(I[:,:,0],cmap = 'Greys_r');
ax[1].imshow(I[:,:,1],cmap = 'Greys_r');
ax[2].imshow(I[:,:,2],cmap = 'Greys_r');


# In[11]:


J = np.zeros((I.shape[2], len(mask[mask!=0]))) 

# Taking every img angle in I and saving pixels that are non-zero
for i in range(0, I.shape[2]):
    # Extract only pixels in the mask
    # Save as i in J
    J[i] = I[:,:,i][mask!=0]


# In[12]:


# Obtain the albedo modulated normal field as M = S^−1J. 
S_inv = np.linalg.inv(S)
M = np.dot(S_inv, J)


# In[13]:


# Normalizing albedo
albedo = np.linalg.norm(M, axis=0)

#norm = np.linalg.norm(M)
#normal_array = an_array/norm
#print(normal_array)

# All the values should be between 0 and 1
print(albedo)


# In[14]:


# Finding albedo within the mask
albedo_mask = np.zeros(mask.shape)
albedo_mask[mask!=0] = albedo


# In[15]:


# Then extract the normal field by normalizing M
# Extract its components n1, n2, n3. Solve for depth and display it at different view points.
# Comment on what happens here. 
normal = (1/ np.linalg.norm(M, axis=0))*M

c1, c2, c3 = normal


# In[16]:


n1 = np.zeros(mask.shape)
n1[mask!=0] = c1

n2 = np.zeros(mask.shape)
n2[mask!=0] = c2

n3 = np.zeros(mask.shape)
n3[mask!=0] = c3


# In[17]:


# help(utils.simchony_integrate)
# n1, n2, n3: nympy float arrays the 3 components of the normal. They must be 2D arrays
# Copying the mask
z = utils.simchony_integrate(n1, n2, n3, mask)
utils.display_surface(z, albedo=None)


# In[ ]:

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
                
                
n1 = normals[:,:,0]
n2 = normals[:,:,1]
n3 = normals[:,:,2]

z = utils.simchony_integrate(n1, n2, n3, mask)
utils.display_surface(z, albedo=None)               
                
smoothed = utils.smooth_normal_field(n1, n2, n3, mask)
n1_s , n2_s, n3_s = smoothed

z = utils.simchony_integrate(n1_s , n2_s, n3_s, mask)
utils.display_surface(z, albedo=None)


utils.display_surface(z, albedo=None)


# help(utils.smooth_normal_field)
smoothed_normals = utils.smooth_normal_field(n1, n2, n3, mask, iters=1000)

n1_s , n2_s, n3_s = smoothed_normals


z = utils.simchony_integrate(n1_s, n2_s, n3_s, mask)
utils.display_surface(z, mask)




