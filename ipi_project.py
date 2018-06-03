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

    def preprocess_sinogram(sinogram, avg_flatfield, darkfield, verbose=False):
        # Flat field corrections
        sinogram = np.log((avg_flatfield - darkfield)/(sinogram - darkfield))

        # Center of mass corrections
        row_sum = np.sum(sinogram, axis=0)
        com = center_of_mass(row_sum)[0]

        x, y = sinogram.shape
        true_com = y / 2.0

        shift_pixel = int(round(true_com - com))

        if verbose:
            print("Sinogram shape: ", sinogram.shape)

        min_delta = 10000.0
        best_k = -1

        # Assume we're at most 20 pixel away from centor of mass.
        for k in range(1, 20):
            com_k = sinogram[:,:-k].shape[1] / 2.0
            calc_com_k = center_of_mass(sinogram[:,:-k])[1]

            delta = np.abs(com_k - calc_com_k)

            if delta < min_delta:
                min_delta = delta
                best_k = k

            if verbose:
                print("k = {}, com_k = {} calc_com_k = {}, delta = {}".format(
                    k, com_k, calc_com_k, delta))

        sinogram = sinogram[:,:-best_k]

        new_row_sum = np.sum(sinogram, axis=0)
        new_com = center_of_mass(new_row_sum)[0]

        if verbose:
            print("True COM ", true_com)
            print("Scan COM", com)
            print("Corrected", new_com)
            print("New True COM", sinogram.shape[1] / 2)
            print("New sino shape", sinogram.shape)

        return sinogram

    avg_flatfield = load_averaged_flatfield()
    darkfield = load_darkfield()
    
    preprocessed_sinogram = preprocess_sinogram(sinogram[:-1], avg_flatfield, darkfield, verbose=True)
        
    # Last scan angle == first scan angle, so we drop the last scan
    return preprocessed_sinogram, images


def load_cached_sinogram(filename):
    return np.load(filename)



def create_OpTomo(size, proj_size, proj_angles):
    #size is the image size (square)
    #proj_size is the proj_size. To avoid problems with backprojection 
    #take the proj_size to be at least sqrt(2)*size
    
    vol_geom = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    
    # For CPU-based algorithms, a "projector" object specifies the projection
    # model used. In this case, we use the "strip" model.
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)


    W = astra.OpTomo(proj_id)

    return W, proj_id



def astra_proj_matrix(size, proj_size, proj_angles):

    W, proj_id = create_OpTomo(size, proj_size, proj_angles)
    
    # Generate the projection matrix for this projection model.
    # This creates a matrix W where entry w_{i,j} corresponds to the
    # contribution of volume element j to detector element i.
    matrix_id = astra.projector.matrix(proj_id)

    # Get the projection matrix as a Scipy sparse matrix.
    A = astra.matrix.get(matrix_id)
    astra.projector.delete(proj_id)
    astra.matrix.delete(matrix_id)

    return A


    



def dist(img_a, img_b):

    norm = np.linalg.norm(img_a - img_b)

    return norm


def reconstruct_image_sirt(proj_angles, sinogram, n_iter, show_reconstruction=False):

    # Warning
    # The geometry is hardcoded for the wooden sample.
    
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

    if show_reconstruction:
        plt.imshow(reconstruction, cmap='gray')
        plt.show()

    return reconstruction



if __name__ == '__main__':

    now = '{date:%Y_%m_%d_%H_%M_%S}.txt'.format( date=datetime.datetime.now() )

    scans_directory = '/mnt/datasets1/fgustafsson/cwi_ct_scan/wooden_block/'

    x, y = get_scan_image_dimension(scans_directory)

    # Load sinogram from the slice in the middle 
    sinogram, images = load_images(scans_directory, slice_at=300)


    # Only use projection in range [0, \pi)
    max_angle = np.pi
    sinogram = sinogram[0:sinogram.shape[0]//2]
    
    
    np.save("wooden_sinogram.npy", sinogram)


    first_image = skimage.io.imread(images[0])


    scanned_angles = sinogram.shape[0]
    sinogram = sinogram[0:scanned_angles]
    scan_width = sinogram.shape[1]
    
    n_iter = 600
    proj_angles = np.linspace(0, (scanned_angles-1)*max_angle / scanned_angles, scanned_angles)


    experiments.tv_inpaint_sinogram(sinogram, proj_angles, drop_rate=0.95)
    experiments.one_angle_reconstruction(sinogram, 42, verbose=True)
    
    experiments.angle_selection_experiment(sinogram, proj_angles, max_angles=20,                                           
                                           iterations=10)
    
    #experiments.greedy_angleselection(sinogram, proj_angles)
    experiments.sinogram_degredation(sinogram, proj_angles)
    experiments.best_sparse_approximation(sinogram, proj_angles, sparse_size=15, trials=250)
    

