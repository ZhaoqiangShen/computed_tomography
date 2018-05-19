import numpy as np
import os
import glob
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import re
import astra
import pylab
from scipy.ndimage import center_of_mass
import datetime

import skimage.filters
import experiments

from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import inpaint_biharmonic
from cvxpy import *

def get_scan_image_dimension(directory):
    one_scan = glob.glob(directory + '/scan*.tif')[0]
    return skimage.io.imread(one_scan).shape


def load_images(directory, slice_at):
    images = glob.glob(directory + '/scan*.tif')
    # Make sure we process the files in the same order they where scanned.
    images.sort(key=lambda f: int(re.search(r'scan_([0-9]+).tif', f).group(1)))

    first_image = skimage.io.imread(images[0])
    x, y = first_image.shape
    sinogram = np.zeros((len(images), y))

    for i, imagefile in enumerate(images):
        image = skimage.io.imread(imagefile)
        sinogram[i] = image[slice_at]

    def load_averaged_flatfield():
        flatfield_images = glob.glob(directory + 'io*.tif')
        avg_flatfield = np.zeros((1, y))

        for imagefile in flatfield_images:
            image = skimage.io.imread(imagefile)
            avg_flatfield += image[slice_at]

        avg_flatfield = avg_flatfield / len(flatfield_images)
        # Extract middle
        return avg_flatfield

    def load_darkfield():
        return skimage.io.imread(os.path.join(directory, 'di000000.tif'))[slice_at]

    def preprocess_sinogram(sinogram, avg_flatfield, darkfield):
        # Flat field corrections
        sinogram = np.log((avg_flatfield - darkfield)/(sinogram - darkfield))

        # Center of mass corrections
        row_sum = np.sum(sinogram, axis=0)
        com = center_of_mass(row_sum)[0]

        x, y = sinogram.shape
        true_com = y / 2.0

        shift_pixel = int(round(true_com - com))

        print("Sinogram shape: ", sinogram.shape)

        min_delta = 10000.0
        best_k = -1
        
        for k in range(1, 20):
            com_k = sinogram[:,:-k].shape[1] / 2.0
            calc_com_k = center_of_mass(sinogram[:,:-k])[1]

            delta = np.abs(com_k - calc_com_k)

            if delta < min_delta:
                min_delta = delta
                best_k = k
            
            print("k = {}, com_k = {} calc_com_k = {}, delta = {}".format(
                k, com_k, calc_com_k, delta))
                                                        
            

        sinogram = sinogram[:,:-best_k]
        #sinogram = np.roll(sinogram, shift_pixel, axis=1)



        new_row_sum = np.sum(sinogram, axis=0)
        new_com = center_of_mass(new_row_sum)[0]


        print("True COM ", true_com)
        print("Scan COM", com)
        print("Corrected", new_com)
        print("New True COM", sinogram.shape[1] / 2)
        print("New sino shape", sinogram.shape)
        return sinogram

    avg_flatfield = load_averaged_flatfield()
    darkfield = load_darkfield()
    
    preprocessed_sinogram = preprocess_sinogram(sinogram[:-1], avg_flatfield, darkfield)
        
    # Last scan angle == first scan angle, so we drop the last scan
    return preprocessed_sinogram, images


def load_cached_sinogram(filename):
    return np.load(filename)


def make_all_same(sinogram, index):
    # Create a new sinogram with just the value sinogram[index] repated
    tiled_measurment = np.tile(sinogram[index], (sinogram.shape[0],1))

    return tiled_measurment


def drop_and_biharmonic_inpaint(drop_rate, sinogram):
    import ipdb; ipdb.set_trace()
    sinogram_mask = np.ones_like(sinogram)

    known_projections = np.random.choice(x, round((1.0 - drop_rate) * x), replace=False)

    # Known pixels marked by zero.
    sinogram_mask[known_projections] = 0.0

    plt.imshow(sinogram_mask)
    plt.show()

    inpaint_biharmonic_sinogram = inpaint_biharmonic(sinogram, sinogram_mask)
    plt.imshow(inpaint_biharmonic_sinogram)
    plt.show()
    return inpaint_biharmonic_sinogram


def drop_and_restore_sinogram(drop_rate, sinogram):
    x, y = sinogram.shape
    U = Variable(x, y)


    known = np.zeros_like(sinogram)

    known_projections = np.random.choice(x, round((1.0 - drop_rate) * x), replace=False)
    
    known[known_projections] = sinogram[known_projections]

    plt.imshow(known, cmap='gray')
    plt.show()

    obj = Minimize(tv(U))
    constraints = [mul_elemwise(known, U) == mul_elemwise(known, sinogram)]
    prob = Problem(obj, constraints)
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=SCS, max_iters=500)

    np.save("U_value.npy", U.value)
    import ipdb; ipdb.set_trace()
    return U.value, known_projections

def dist(img_a, img_b):

    norm = np.linalg.norm(img_a - img_b)

    return norm


def reconstruct_image_sirt(proj_angles, sinogram, n_iter):

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

    scan_width = sinogram.shape[1]
    vol_geom = astra.create_vol_geom(scan_width, scan_width)
    proj_geom = astra.create_proj_geom('fanflat', d_p, scan_width, proj_angles, SOD_p, ODD_p)
    #proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)

    reconstruction = W.reconstruct('SIRT_CUDA',
                                   sinogram,
                                   iterations=n_iter,
                                   extraOptions={'MinConstraint': 0.0})

    #plt.imshow(reconstruction, cmap='gray')
    #plt.show()

    return reconstruction



if __name__ == '__main__':

    now = '{date:%Y_%m_%d_%H_%M_%S}.txt'.format( date=datetime.datetime.now() )

    scans_directory = '/mnt/datasets1/fgustafsson/cwi_ct_scan/wooden_block/'

    x, y = get_scan_image_dimension(scans_directory)

    # Load sinogram from the slice in the middle 
    sinogram, images = load_images(scans_directory, slice_at=300)

    max_angle = np.pi

    sinogram = sinogram[0:sinogram.shape[0]//2]
    
    
    np.save("wooden_sinogram.npy", sinogram)

    #sinogram = load_cached_sinogram("wooden_sinogram.npy")

    #plt.imshow(sinogram, cmap='gray')
    #plt.show()

    # As a sanity check look at first scan at angle 0
    first_image = skimage.io.imread(images[0])
    #plt.imshow(first_image, cmap='gray')
    #plt.show()

    scanned_angles = sinogram.shape[0]
    sinogram = sinogram[0:scanned_angles]
    scan_width = sinogram.shape[1]


    
    n_iter = 600
    proj_angles = np.linspace(0, (scanned_angles-1)*max_angle / scanned_angles, scanned_angles)

    #sinogram_restored, kept_angles = drop_and_restore_sinogram(0.98, sinogram)

    #proj_angles_subset = proj_angles[kept_angles]
    #sinogram_subset = sinogram[kept_angles]


    experiments.angle_selection_experiment(sinogram, proj_angles, max_angles=20,                                           
                                           iterations=10)
    
    #experiments.greedy_angleselection(sinogram, proj_angles)
    #exit(0)
    #experiments.best_sparse_approximation(sinogram, proj_angles, sparse_s=15)
    #experiments.sinogram_degredation(sinogram, proj_angles)
    #exit(0)

    
    #print("Kept {} out of {} projections".format(kept_angles.shape, proj_angles.shape))


    #biharmonic_reconstructed_sinogram = drop_and_biharmonic_inpaint(0.90, sinogram)
    
    all_same_sinogram = make_all_same(sinogram, 100)
    #all_same_reconstruction = reconstruct_image_sirt(proj_angles, all_same_sinogram, n_iter)


    one_angle_reconstruction = reconstruct_image_sirt(
        proj_angles[1:100],
        sinogram[1:100],
        n_iter)
    
    tv_reconstruction = reconstruct_image_sirt(proj_angles, sinogram_restored, n_iter)
    limited_reconstruction = reconstruct_image_sirt(proj_angles_subset, sinogram_subset, n_iter)
    orignal_reconstruction = reconstruct_image_sirt(proj_angles, sinogram, n_iter)
    
    denoised = denoise_tv_chambolle(limited_reconstruction, weight=0.0001)
    
    plt.figure(figsize = (30,10))
    plt.subplot(141)
    plt.imshow(orignal_reconstruction, cmap='gray')
    plt.subplot(142)
    plt.imshow(limited_reconstruction, cmap='gray')
    plt.subplot(143)
    plt.imshow(denoised, cmap='gray')
    plt.subplot(144)
    plt.imshow(np.abs(orignal_reconstruction - denoised), cmap='gray')
    plt.show()

    exit(0)

    #projection_subset = np.random.choice(scanned_angles, round(0.9 * scanned_angles), replace=False)
    #import ipdb; ipdb.set_trace()
    #proj_angles = proj_angles[projection_subset]
    #sinogram = sinogram[projection_subset]
    
    #sinogram[projection_subset] = np.min(sinogram)
    #plt.imshow(sinogram, cmap='gray')
    #plt.show()

