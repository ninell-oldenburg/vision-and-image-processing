#!/usr/bin/env python
# coding: utf-8

# ### Exercise 2

# mat vase is a synthetic and clean dataset, with exactly 3 images

# In[4]:


import ps_utils as utils
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


filename = "Beethoven.mat"
path = os.path.join("utils", filename)

# Reading shiny vase matlab file
# I - 3D array of image size (m,n,k) where k is views
# mask - with records of intensity data
# S - light vectors 
I_beet, mask_beet, S_beet = utils.read_data_file(path)


# In[7]:


# show the 3 available images of beethoven
fig, axes = plt.subplots(1, 3)
ax = axes.ravel()

ax[0].imshow(I_beet[:,:,0],cmap = 'Greys_r');
ax[1].imshow(I_beet[:,:,1],cmap = 'Greys_r');
ax[2].imshow(I_beet[:,:,2],cmap = 'Greys_r');


# In[9]:


# show what is masked
imshow(mask_beet, cmap = 'Greys_r');


# In[10]:


# show beethoven clipped with the mask
imshow(mask_beet * I_beet[:,:,0]);


# In[11]:


mask_beet_new = 1 - mask_beet #take 1-mask so that the face is shown and not the background


# In[12]:


J = np.ndarray((3,mask_beet.sum())) #create nd array
for i in range(I_beet.shape[2]):
    masked_I_beet = np.ma.masked_array(I_beet[:,:,i], mask=mask_beet_new)
    compressed_I_beet = masked_I_beet.compressed()
    J[i] = compressed_I_beet


# In[13]:


J.shape


# In[14]:


# get M= S^-1 J
S_beet_inv = np.linalg.inv(S_beet)
M = S_beet_inv @J


# In[15]:


# With it(M), extract the albedo
# within the mask, display it as a 2D image.


# In[16]:


# calculate albedo
norm_M_beet = np.linalg.norm(M, axis = 0)
albedo_beet = norm_M_beet


# In[17]:


# convert back to image
# inspired by https://stackoverflow.com/questions/38855058/inverting-the-numpy-ma-compressed-operation
albedo_image_beet = np.ndarray(mask_beet.shape)
np.place(albedo_image_beet,masked_I_beet.mask,0)
np.place(albedo_image_beet,~masked_I_beet.mask,albedo_beet)


# In[19]:


# display the albedo 
imshow(albedo_image_beet,cmap = 'Greys_r');


# In[20]:


# CALCULATE normal
normal = (1/albedo_beet)*M
n1,n2,n3 = normal #unpack to seperate variables


# In[21]:


# convert back to 256x256 array
# inspired by https://stackoverflow.com/questions/38855058/inverting-the-numpy-ma-compressed-operation
norm1 = np.ndarray(mask_beet.shape)
np.place(norm1,masked_I_beet.mask,0)
np.place(norm1,~masked_I_beet.mask,n1)


# In[22]:


norm2 = np.ndarray(mask_beet.shape)
np.place(norm2,masked_I_beet.mask,0)
np.place(norm2,~masked_I_beet.mask,n2)


# In[23]:


norm3 = np.ndarray(mask_beet.shape)
np.place(norm3,masked_I_beet.mask,0)
np.place(norm3,~masked_I_beet.mask,n3)


# In[24]:


z_beet = utils.simchony_integrate(norm1,norm2,norm3, mask_beet)


# In[20]:


utils.display_surface(z_beet, albedo=None)

