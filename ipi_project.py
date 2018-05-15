import numpy as np
import os
import glob
import skimage.io
import matplotlib.pyplot as plt
import re
import astra
import pylab
from scipy.ndimage import center_of_mass


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
        
        sinogram = np.roll(sinogram, shift_pixel, axis=1)

        new_row_sum = np.sum(sinogram, axis=0)
        new_com = center_of_mass(new_row_sum)[0]

        print("True COM ", true_com)
        print("Scan COM", com)
        print("Corrected", new_com)
        return sinogram

    avg_flatfield = load_averaged_flatfield()
    darkfield = load_darkfield()
    
    preprocessed_sinogram = preprocess_sinogram(sinogram[:-1], avg_flatfield, darkfield)
        
    # Last scan angle == first scan angle, so we drop the last scan
    return preprocessed_sinogram, images


def load_cached_sinogram(filename):
    return np.load(filename)


if __name__ == '__main__':

    scans_directory = '/mnt/datasets1/fgustafsson/cwi_ct_scan/wooden_block/'

    x, y = get_scan_image_dimension(scans_directory)

    # Load sinogram from the slice in the middle 
    sinogram, images = load_images(scans_directory, slice_at=x // 2)
    np.save("wooden_sinogram.npy", sinogram)

    #sinogram = load_cached_sinogram("wooden_sinogram.npy")

    plt.imshow(sinogram, cmap='gray')
    plt.show()

    # As a sanity check look at first scan at angle 0
    first_image = skimage.io.imread(images[0])
    plt.imshow(first_image, cmap='gray')
    plt.show()

    scanned_angles = sinogram.shape[0]
    scan_width = sinogram.shape[1]

    n_iter = 100
    vol_geom = astra.create_vol_geom(scan_width, scan_width)
    proj_angles = np.linspace(0, (scanned_angles-1)*2.0*np.pi / scanned_angles, scanned_angles)

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
    #f_inv  = W.reconstruct('SIRT_CUDA', sinogram, iterations=n_iter,extraOptions={'MinConstraint':0.0})
    f_inv  = W.reconstruct('FBP', sinogram, extraOptions={'MinConstraint':0.0})
    plt.figure(figsize = (10,10))
    plt.imshow(f_inv, cmap='gray')
    plt.imsave("reconstructed.png", f_inv, cmap='gray')
    plt.show()
    
    #Testcomment Francien
