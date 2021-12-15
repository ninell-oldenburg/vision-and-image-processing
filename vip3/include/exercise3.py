#!/usr/bin/env python
# coding: utf-8

# ### Exercise 3

# mat vase is a synthetic and clean dataset, with exactly 3 images

# In[2]:


import ps_utils as utils
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


filename = "mat_vase.mat"
path = os.path.join("utils", filename)

# Reading shiny vase matlab file
# I - 3D array of image size (m,n,k) where k is views
# mask - with records of intensity data
# S - light vectors 
I_vase, mask_vase, S_vase = utils.read_data_file(path)


# In[6]:


# show the 3 available images of the vase
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()

ax[0].imshow(I_vase[:,:,0],cmap = 'Greys_r')
ax[1].imshow(I_vase[:,:,1],cmap = 'Greys_r')
ax[2].imshow(I_vase[:,:,2],cmap = 'Greys_r')


# In[8]:


# show what is masked
imshow(mask_vase, cmap = 'Greys_r');


# In[9]:


# show vase clipped with the mask
imshow(mask_vase * I_vase[:,:,0]);


# In[10]:


mask_vase_new = 1 - mask_vase #take 1-mask so that the vase is shown and not the background


# In[11]:


J = np.ndarray((3,mask_vase.sum())) #create nd array sum() to get the amount of non-zero cells
for i in range(I_vase.shape[2]):
    masked_I_vase = np.ma.masked_array(I_vase[:,:,i], mask=mask_vase_new)
    compressed_I_vase = masked_I_vase.compressed()
    J[i] = compressed_I_vase


# In[12]:


J.shape


# In[13]:


# get M= S^-1 J
S_vase_inv = np.linalg.inv(S_vase)
M = S_vase_inv @J


# In[14]:


M.shape


# In[15]:


# With it(M), extract the albedo
# within the mask, display it as a 2D image.


# In[16]:


# calculate albedo
norm_M_vase = np.linalg.norm(M, axis = 0)
norm_M_vase.shape
albedo_vase = norm_M_vase


# In[17]:


mask_vase.shape


# In[19]:


# convert back to image
# inspired by https://stackoverflow.com/questions/38855058/inverting-the-numpy-ma-compressed-operation
albedo_image_vase = np.ndarray(mask_vase.shape)
np.place(albedo_image_vase,masked_I_vase.mask,0)
np.place(albedo_image_vase,~masked_I_vase.mask,albedo_vase)


# In[20]:


# display the albedo 
imshow(albedo_image_vase,cmap = 'Greys_r');


# In[21]:


# CALCULATE normal
normal = (1/albedo_vase)*M


# In[22]:


normal.shape


# In[23]:


n1,n2,n3 = normal #unpack to seperate variables


# In[24]:


# convert back to 256x256 array
# inspired by https://stackoverflow.com/questions/38855058/inverting-the-numpy-ma-compressed-operation
norm1 = np.ndarray(mask_vase.shape)
np.place(norm1,masked_I_vase.mask,0)
np.place(norm1,~masked_I_vase.mask,n1)


# In[25]:


norm2 = np.ndarray(mask_vase.shape)
np.place(norm2,masked_I_vase.mask,0)
np.place(norm2,~masked_I_vase.mask,n2)


# In[26]:


norm3 = np.ndarray(mask_vase.shape)
np.place(norm3,masked_I_vase.mask,0)
np.place(norm3,~masked_I_vase.mask,n3)


# In[27]:


z_vase = utils.simchony_integrate(norm1,norm2,norm3, mask_vase)


# In[24]:


utils.display_surface(z_vase, albedo=None)

