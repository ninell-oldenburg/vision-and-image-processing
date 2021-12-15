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
import cv2
import utils.ps_utils as utils
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


# In[ ]:


mask.astype(np.uint8)


# In[ ]:


#fig, axes = plt.subplots(1, 10)
#ax = axes.ravel()

# images = data.stereo_motorcycle()
#ax[0].imshow(I[:,:,0],cmap = 'Greys_r');
#ax[1].imshow(I[:,:,1],cmap = 'Greys_r');
#ax[2].imshow(I[:,:,2],cmap = 'Greys_r');
#ax[3].imshow(I[:,:,3],cmap = 'Greys_r');
#ax[4].imshow(I[:,:,4],cmap = 'Greys_r');
#ax[5].imshow(I[:,:,5],cmap = 'Greys_r');
#ax[6].imshow(I[:,:,6],cmap = 'Greys_r');
#ax[7].imshow(I[:,:,7],cmap = 'Greys_r');
#ax[8].imshow(I[:,:,8],cmap = 'Greys_r');
#ax[9].imshow(I[:,:,9],cmap = 'Greys_r');


# In[ ]:


plt.imshow(mask, cmap = 'Greys_r')


# In[ ]:


plt.imshow(mask * I[:,:,0], cmap= 'Greys_r')


#     If nz is the number of pixels inside the non-zero part of the mask, You should create an array J of size/shape (10, nz)  and obtain the albedo modulated normal field as M = S^|J (the pseudo-inverse). With it, extract the albedo within the mask, display it as a 2D image. Then extract the normal field by normalizing M ,extract its components n1, n2, n3. Solve for depth and display it at different view points. Comment on what happens here. Do you think that RANSAC could provide a better estimation of normals? Explain. 
#     
#     You should try and replace Woodham’s first step (via inverse/pseudoinverse) with RANSAC estimation. The threshold parameter in ransac 3dvector() should no more than be 2.0. After the estimation for each pixel, extract the normals and the albedo. Display and comment on the results. Do they differ from Woodham’s estimation? Try then to make the estimated normal field smoother using the smooth normal field() function. You may experiment with the iters parameter.

# In[ ]:


# Matrix and vector approach

# If nz is the number of pixels inside the non-zero part of the mask, You should create an array J of size/shape (10, nz)
J = np.ndarray((I.shape[2], len(mask[mask!=0]))) 
J.shape

# Taking every img angle in I and saving pixels that are non-zero
for i in range(0, I.shape[2]):
    # Extract only pixels in the mask
    # Save as i in J
    J[i] = I[:,:,i][mask!=0]
    
# Extracting m with the pseudo inverse function
M = np.linalg.pinv(S)@J
albedo = np.linalg.norm(M, axis=0)

# Finding albedo within the mask
albedo_mask = np.zeros(mask.shape)
albedo_mask[mask!=0] = albedo

# Calculating normals
normal = (1/ np.linalg.norm(M, axis=0))*M

# Componenets
c1, c2, c3 = normal

# Building matrix from mask and normals
n1 = np.zeros(mask.shape)
n1[mask!=0] = c1

n2 = np.zeros(mask.shape)
n2[mask!=0] = c2

n3 = np.zeros(mask.shape)
n3[mask!=0] = c3


# In[ ]:


# Pixel by pixel approach

M = []
albedo = []
normal = []

# Per row in image
for i in range(0,I.shape[0]):
    # idx += 1
    # Per column in image
    for j in range(0, I.shape[1]):
        # If the position in the mask is non-zero
        # Apply RANSAC to the image pixel
        if mask[i,j]!=0:
            j = I[i,j,:]
            m = np.linalg.pinv(S)@j
            p = np.linalg.norm(m)
            n = (1/ np.linalg.norm(m))*m
            
            albedo.append(p)
            M.append(m)
            normal.append(n)
            
normal = np.array(normal).reshape(3, 49872)           
M = np.array(M).reshape(3, 49872)
albedo = np.array(albedo)

c1, c2, c3 = normal

n1 = np.zeros(mask.shape)
n1[mask!=0] = c1

n2 = np.zeros(mask.shape)
n2[mask!=0] = c2

n3 = np.zeros(mask.shape)
n3[mask!=0] = c3


# In[ ]:


# help(utils.simchony_integrate)
# n1, n2, n3: nympy float arrays the 3 components of the normal. They must be 2D arrays
# Copying the mask
z = utils.simchony_integrate(n1, n2, n3, mask)
z_unbiased = utils.unbiased_integrate(n1, n2, n3, mask)


# In[ ]:


z[np.isnan(z)==False]


# In[ ]:


# help(utils.display_surface)
#utils.display_surface(z, albedo=albedo_mask)


# In[ ]:


#![algorithm_ps.png](attachment:algorithm_ps.png)


#     Do you think that RANSAC could provide a better estimation of normals? Explain. You should try and replace Woodham's first step (via inverse/pseudoinverse) with RANSAC estimation. The threshold parameter in ransac_3dvector() should no more than be 2.0. 

#     When using RANSAC, however, an estimation of albedo and normal will have to be performed one pixel at a time.

#     After the estimation for each pixel, extract the normals and the albedo. Display and comment on the results. Do they differ from Woodham's estimation? 

# In[ ]:


ransac = []
albedo = []
normal = []

# Per row in image
for i in range(0,I.shape[0]):
    # Per column in image
    for j in range(0, I.shape[1]):
        # If the position in the mask is non-zero
        # Apply RANSAC to the image pixel
        if mask[i,j]!=0:
            m, inliers, best_fit = utils.ransac_3dvector(data=(I[i,j,:], S), threshold = 25.0)
            p = np.linalg.norm(m, axis=0)
            n = (1/ np.linalg.norm(m, axis=0))*m
            
            albedo.append(p)
            ransac.append(m)
            normal.append(n)
            
normal = np.array(normal).reshape(3, 24828)           
ransac = np.array(ransac).reshape(3, 24828)
albedo = np.array(albedo)

c1, c2, c3 = normal

n1 = np.zeros(mask.shape)
n1[mask!=0] = c1

n2 = np.zeros(mask.shape)
n2[mask!=0] = c2

n3 = np.zeros(mask.shape)
n3[mask!=0] = c3


# In[ ]:


# n1, n2, n3: nympy float arrays the 3 components of the normal. They must be 2D arrays
# Copying the mask
z = utils.simchony_integrate(n1, n2, n3, mask)
# z_unbiased = utils.unbiased_integrate(n1, n2, n3, mask)


# In[ ]:


z[np.isnan(z)==False]


# In[ ]:


utils.display_surface(z, albedo=albedo_mask)


#     Try then to make the estimated normal field smoother using the smooth normal field() function. You may experiment with the iters parameter.

# In[ ]:


# Smoothing the normal field
help(utils.smooth_normal_field)
smoothed_normals = utils.smooth_normal_field(n1, n2, n3, mask)

n1_s , n2_s, n3_s = smoothed_normals

# Solving for depth and displaying the image
z = utils.simchony_integrate(n1_s, n2_s, n3_s, mask)
utils.display_surface(z, albedo=albedo_mask)


# In[ ]:




