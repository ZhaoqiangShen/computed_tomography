#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:35:32 2018

@author: bossema
"""
import numpy as np
import matplotlib.pyplot as plt
import astra

def load_cached_sinogram(filename):
    return np.load(filename)


if __name__ == '__main__':

#    scans_directory = '/mnt/datasets1/fgustafsson/cwi_ct_scan/wooden_block/'
#
#    x, y = get_scan_image_dimension(scans_directory)
#
#    # Load sinogram from the slice in the middle 
#    sinogram, images = load_images(scans_directory, slice_at=x // 2)
#    np.save("wooden_sinogram.npy", sinogram)

    sinogram = load_cached_sinogram("wooden_sinogram.npy")

    plt.imshow(sinogram, cmap='gray')
    plt.show()

    # As a sanity check look at first scan at angle 0
#    first_image = skimage.io.imread(images[0])
#    plt.imshow(first_image, cmap='gray')
#    plt.show()
    scanned_angles = sinogram.shape[0]
    proj_angles = np.linspace(0, (scanned_angles-1)*2.0*np.pi / scanned_angles, scanned_angles)
    n_iter = 100

def reconstruct_SIRT(sinogram,n_iter,proj_angles):

    scan_width = sinogram.shape[1]
    vol_geom = astra.create_vol_geom(scan_width, scan_width)

    # Scan geometry
    SDD = 498.0
    SOD = 313.001465
    d = 0.149600

    # Apply scaling to match Astras pixelsize v = 1
    v = d * SOD / SDD
    SDD_p = SDD / v
    SOD_p = SOD / v
    ODD_p = SDD_p - SOD_p
    d_p = d / v

    proj_geom = astra.create_proj_geom('fanflat', d_p, scan_width, proj_angles, SOD_p, ODD_p)
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)
    f_inv  = W.reconstruct('SIRT_CUDA', sinogram, iterations=n_iter,extraOptions={'MinConstraint':0.0})
    #f_inv  = W.reconstruct('FBP', sinogram, extraOptions={'MinConstraint':0.0})
    plt.figure(figsize = (10,10))
    plt.imshow(f_inv, cmap='gray')
    plt.imsave("reconstructed.png", f_inv, cmap='gray')
    plt.show()
    
    
def angle_subset_reconstruction(sinogram, n_iter, proj_angles):
    #input the projection angles in degrees
    n_angles = len(proj_angles)
    scan_width = sinogram.shape[1]
    
    sinogram_subset = np.zeros((n_angles, scan_width))
    for i in range(n_angles):
        sinogram_subset[i,:] = sinogram[int(proj_angles[i]/360*1800),:] #or should we subtract 1?

    proj_angles = proj_angles/180*np.pi
    
    plt.imshow(sinogram_subset, cmap='gray')
    plt.show()
    
    reconstruct_SIRT(sinogram_subset,n_iter,proj_angles)
   
#%%
#equispaced, 10 angles
proj_angles = np.linspace(0, 180,10)
angle_subset_reconstruction(sinogram, n_iter, proj_angles)
#%%
#10 angles very close to 0
proj_angles = np.linspace(0, 10,10)
angle_subset_reconstruction(sinogram, n_iter, proj_angles)

#%%
#equispaced, 40 angles
proj_angles = np.linspace(0, 180,40)
angle_subset_reconstruction(sinogram, n_iter, proj_angles)