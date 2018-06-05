# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:41:54 2018

@author: Francien
"""
import matplotlib.pyplot as plt 
import numpy as np
import astra
import scipy

def astra_proj_matrix(size, proj_size, proj_angles):
    #size is the image size (square)
    #proj_size is the proj_size. To avoid problems with backprojection 
    #take the proj_size to be at least sqrt(2)*size
    
    vol_geom = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    
    # For CPU-based algorithms, a "projector" object specifies the projection
    # model used. In this case, we use the "strip" model.
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)

    # Generate the projection matrix for this projection model.
    # This creates a matrix W where entry w_{i,j} corresponds to the
    # contribution of volume element j to detector element i.
    matrix_id = astra.projector.matrix(proj_id)

    # Get the projection matrix as a Scipy sparse matrix.
    A = astra.matrix.get(matrix_id)
    astra.projector.delete(proj_id)
    astra.matrix.delete(matrix_id)
    return(A)

A = astra_proj_matrix(32, 48, np.array([0,90,120])/180*np.pi)
svd = scipy.sparse.linalg.svds(A, k = min(A.shape)-1)
values = sorted(svd[1], reverse = True)
y = range(min(A.shape)-1)
plt.scatter(y,values)
plt.title('Singular values of matrix A')
