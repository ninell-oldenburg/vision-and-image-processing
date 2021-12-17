#!/usr/bin/env python
# coding: utf-8

# ### Exercise 7

# Buddha is real dataset, with exactly 27 images

# In[8]:


import ps_utils as utils
import os
from skimage.io import imshow
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


filename = "face.mat"

path = os.path.join("utils", filename)
# Reading face matlab file
# I - 3D array of image size (m,n,k) where k is views
# mask - with records of intensity data
# S - light vectors 
I_face, mask_face, S_face = utils.read_data_file(path)


# In[13]:


# show the first 5 available images of the face
fig, axes = plt.subplots(1, 5)
ax = axes.ravel()

for i in range(5):
    ax[i].imshow(I_face[:,:,i],cmap = 'Greys_r')


# In[14]:


# show what is masked, which is nothing.
imshow(mask_face, cmap = 'Greys_r');


# In[15]:


mask_face.sum() == mask_face.size #no part of the image is masked. so the masked image is identical to the unmasked image


# In[16]:


# show face clipped with the mask
imshow(mask_face * I_face[:,:,0]);


# In[17]:


mask_face_new = 1 - mask_face #take 1-mask so that the face is shown and not the background


# In[18]:


nr_of_pictures = I_face.shape[2]


# In[19]:


J = np.ndarray((nr_of_pictures,mask_face.size)) #create nd array with the the amount of non-zero cells
for i in range(nr_of_pictures):
    masked_I_face = np.ma.masked_array(I_face[:,:,i], mask=mask_face_new)
    compressed_I_face = masked_I_face.compressed()
    J[i] = compressed_I_face


# In[20]:


J.shape


# In[21]:


# implement ransac


# In[ ]:


normal_field = np.zeros((I_face.shape[0], I_face.shape[1],3))
for i in range(I_face.shape[0]):
    for j in range(I_face.shape[1]):
        m, inliers, best_fit = utils.ransac_3dvector(data =(I_face[i,j,:],S_face), threshold =10) 
        normal_field[i,j] = m


# In[14]:


n1 = normal_field[:,:,0]
n2 = normal_field[:,:,1]
n3 = normal_field[:,:,2]


# In[15]:


z_face = utils.unbiased_integrate(n1,n2,n3, mask_face)
utils.display_surface(z_face, albedo=None) #show non smoothed normal field


# In[16]:



smooth_normal= utils.smooth_normal_field(n1, n2, n3,mask_face, iters = 1 )#here to obtain different amount of iterations chagne iter
smooth_n1,smooth_n2,smooth_n3 = smooth_normal #unpack the normal field

z_face_smooth = utils.unbiased_integrate(smooth_n1,smooth_n2,smooth_n3, mask_face)#create depth field
utils.display_surface(z_face_smooth, albedo=None) #show  smoothed normal field

