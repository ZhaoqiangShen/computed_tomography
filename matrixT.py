# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:04:26 2018

@author: Francien
"""

from topolar import topolar
import numpy as np
from phantoms import circle
import pylab
import scipy.sparse

from skimage.filters import gaussian


def create_T(size):

    length = size**2
    T = np.zeros((length,length))

    for i in range(length):
        i_vec = np.zeros(length)
        i_vec[i] = 1
        T[:,i] = topolar(np.reshape(i_vec,(size,size)))[0].ravel()
    return T

def create_Blur(size, sigma):

    length = size**2
    G = np.zeros((length, length))


    for i in range(length):
        i_vec = np.zeros(length)
        i_vec[i] = 1
        G[:,i] = gaussian(i_vec.reshape(size,size),
                          sigma=sigma,
                          preserve_range=True).ravel()
    return G


def create_Dx(size):

    # Column version
    Dx = np.diag([1]*size) -1 * np.diag([1]*(size -1), k=1)
    Dx[-1, 0] = -1.0

    # Operator that take whole image as one vector (size**2)
    Dx_big = scipy.sparse.block_diag([Dx for i in range(size)])

    return Dx_big.T

def create_Mask(size, keep_rows):
    # Keep the first 'keep_rows' in a 2D image,
    # Set rest to zero.

    assert keep_rows <= size

    return scipy.sparse.diags([1]*keep_rows*size + [0]*(size - keep_rows)*size)


if __file__ == '__main__':

#%%
    
    size = 64
    T = create_T(size)
    test = circle(size, 4, wall_thickness = 1)
    pylab.imshow(test) 
    pylab.show()
    test_matrix = np.reshape(T@test.ravel(), (size,size))
    pylab.imshow(test_matrix)
    pylab.title('T@test')
    pylab.show()
    test_topolar=topolar(test)[0]
    pylab.imshow(test_topolar) 
    pylab.title('topolar(test)') 
    pylab.show()
    diff = test_matrix-test_topolar
    pylab.imshow(diff)
    pylab.title('difference')
    pylab.show()
    print(diff.max(),diff.min())


