# -*- coding:utf-8 -*-

import numpy as np
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon, rotate
from skimage.transform._warps_cy import _warp_fast
from skimage.util import crop, pad
from scipy import misc
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片
from multiprocessing.dummy import Pool as ThreadPool
from math import floor, ceil
import sinogram_interpolation as interp
import SplineInterpolation3 as interp3
import scipy.io

def timeit(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return timed

# @timeit
def zero_padding_radon(image, theta, circle=False, plot=True):
    """

    :param image:
    :param theta:
    :param circle:
    :param plot:
    :return:
    """
    sinogram = radon(image, theta, circle=circle)

    # plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image, cmap='gray_r')
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap='gray_r',
                   extent=(0, 160, sinogram.shape[0], 0), aspect='auto')
        fig.tight_layout()
        plt.show()
    return sinogram


def max_coordinate(block_size, original_image_size):

    return np.array(original_image_size) // block_size - 1


def shift_radon_transform(block_coordinate, image_block, original_image_size, theta, circle=False, plot=True):
    """

    :param block_coordinate: coordinate of block. e.g [0, 0] is the upper left block
    :param image_block:
    :param original_image_size:  shape of original image, e.g [160, 160]
    :param theta:
    :param circle:
    :param plot:
    :return:
    """

    block_size = image_block.shape[0]
    sinogram = radon(image_block, theta, circle=circle) # 一个block的投影

    block_movement_v, block_movement_h = (np.array(block_coordinate) + 1) * block_size - block_size // 2 # [10 10]
    H, W = original_image_size # [160 160]
    assert H == W, "Image should be squared."

    T_length = int(np.sqrt(2)*H)+1 # T_length == 227 when image shape is [160, 160]
    shifted_sinogram = np.zeros([T_length, sinogram.shape[1]]) # [227 160] same as ground truth
    shifted_sinogram_interp = np.zeros([T_length, sinogram.shape[1]])
    # distance from center
    block_movement_h_center = H / 2 - block_movement_h # 70
    block_movement_v_center = H / 2 - block_movement_v # 70

    delta_v = -block_movement_h_center
    delta_h = block_movement_v_center

#    position_data = np.zeros(sinogram.shape[1])
    for column_num in range(sinogram.shape[1]): # 逐角度进行偏移 colume range = 160
        # put the sinogram into the same column, but shited rows
        th = (np.pi / 180.0) * theta[column_num]
        
        position_start = float(T_length) / 2 - float(sinogram.shape[0] / 2) + \
                                delta_v * np.cos(th) + delta_h * np.sin(th)
                                
#        position_data[column_num] = position_start
        row_start = int(np.round(position_start))
#        res = position_start - row_start
        
        row_end = row_start + sinogram.shape[0] # 29
        
        shifted_sinogram[row_start: row_end, column_num] = sinogram[:, column_num]
        '''
        linear interpolation
        '''
        shifted_sinogram_interp = interp.sinogram_linear_interpolation(sinogram, shifted_sinogram_interp, column_num, position_start, T_length)
        '''
        2 order spline interpolation
        '''
#        x = np.zeros([sinogram.shape[0],])
##        res = position_start - np.round(position_start)
#        for x_num in range(sinogram.shape[0]):
#            x[x_num] = position_start + x_num
#        y = sinogram[:, column_num]
#        result=interp.solutionOfEquation(interp.calculateEquationParameters(x), x, y)
#        length = sinogram.shape[0]-1
#        
#        for i in range(length):
#            if round(position_start) - position_start >=0:
#                shifted_sinogram_interp[length] = sinogram[length, column_num]
#                if i == 0:
#                    shifted_sinogram_interp[row_start + i, column_num] = \
#                    interp.calculate([0, result[0], result[1]], [i + position_start])[0]
#                else:
#                    shifted_sinogram_interp[row_start + i, column_num] = \
#                    interp.calculate([result[3*i-1], result[3*i], result[3*i+1]], [i + position_start])[0]
#            else:
#                shifted_sinogram_interp[0] = sinogram[0, column_num]
#                if i == 0:
#                    shifted_sinogram_interp[row_start + i + 1, column_num] = \
#                    interp.calculate([0, result[0], result[1]], [i + position_start])[0]
#                else:
#                    shifted_sinogram_interp[row_start + i + 1, column_num] = \
#                    interp.calculate([result[3*i-1], result[3*i], result[3*i+1]], [i + position_start])[0]
#        pos = np.argmin(shifted_sinogram_interp[row_start:row_end,column_num])
#        shifted_sinogram_interp[pos+row_start,column_num] = sinogram[pos,column_num]
        
        '''
        3 order spline interpolation
        '''
#        x = np.zeros([sinogram.shape[0],])
##        res = position_start - np.round(position_start)
#        for x_num in range(sinogram.shape[0]):
#            x[x_num] = position_start + x_num
#        y = sinogram[:, column_num]
#        result=interp3.solutionOfEquation(interp3.calculateEquationParameters(x),x, y)
#        length = sinogram.shape[0]-1
#        
#        for i in range(length):
#            if round(position_start) - position_start >=0:
#                shifted_sinogram_interp[length] = sinogram[length, column_num]
#                if i == 0:
#                    shifted_sinogram_interp[row_start + i, column_num] = \
#                    interp3.calculate([0,0, result[0], result[1]], [i + position_start])[0]
#                else:
#                    shifted_sinogram_interp[row_start + i, column_num] = \
#                    interp3.calculate([result[4*i-2], result[4*i-1], result[4*i], result[4*i+1]], [i + position_start])[0]
#            else:
#                shifted_sinogram_interp[0] = sinogram[0, column_num]
#                if i == 0:
#                    shifted_sinogram_interp[row_start + i + 1, column_num] = \
#                    interp3.calculate([0, 0, result[0], result[1]], [i + position_start])[0]
#                else:
#                    shifted_sinogram_interp[row_start + i + 1, column_num] = \
#                    interp3.calculate([result[4*i-2], result[4*i-1], result[4*i], result[4*i+1]], [i + position_start])[0]
#
#        shift_sinogram = np.zeros(sinogram.shape[0])
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(shifted_sinogram_interp, cmap= 'gray')
#        length = sinogram.shape[0]-1
#        if round(position_start) - position_start >=0:
#            step = round(position_start) - position_start
#            shift_sinogram[:length] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[0:length, column_num]
#            shift_sinogram[length] = sinogram[length, column_num]
#        else:
#            step = 1 - position_start + round(position_start)
#            shift_sinogram[0] = sinogram[0, column_num]
#            shift_sinogram[1:] = (sinogram[1:, column_num] - sinogram[0:length, column_num]) * step + sinogram[0:length, column_num]
#        shifted_sinogram_interp[row_start: row_end, column_num] = shift_sinogram[:]
        
    # plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image_block, cmap='gray_r')
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap='gray_r',
                   extent=(0, 160, 0, sinogram.shape[0]), aspect='auto')
        fig.tight_layout()
        plt.show()

    return shifted_sinogram, shifted_sinogram_interp


def block_iradon(shifted_sinogram,theta):
    """

    :param shifted_sinogram:
    :param theta:
    :return:
    """
    reconstruction_fbp = iradon(shifted_sinogram, theta=theta, circle=False)

    return reconstruction_fbp

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def discrete_radon_transform(image, theta, pad_width):

    padded_image = np.pad(image, pad_width, mode='constant',
                          constant_values=0)
    # padded_image is always square
    assert padded_image.shape[0] == padded_image.shape[1]

    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)))

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    for i in range(len(theta)):
        rotated = _warp_fast(padded_image, build_rotation(theta[i]))
        radon_image[:, i] = rotated.sum(0)
    return radon_image

def non_zero_values_in_column_i(sinogram, column_i):
    print(np.nonzero(sinogram[:, column_i]))
    return sinogram[np.nonzero(sinogram[:, column_i]), column_i]

def show_reconstructed_img(sinogram_zero_padding, shifted_sinogram, block_coordinate, block_size, original_image, theta, save_path=None):
    """

    :param sinogram_zero_padding:
    :param shifted_sinogram:
    :param original_image:
    :param theta:
    :return:
    """
    reconstruction_fbp = iradon(shifted_sinogram, theta=theta, circle=False)
    block_x, block_y = block_coordinate
    reconstruction_shift_radon = np.zeros(original_image.shape)
    reconstruction_shift_radon[block_x*block_size:(block_x+1)*block_size, block_y*block_size:(block_y+1)*block_size] = reconstruction_fbp[block_x*block_size:(block_x+1)*block_size, block_y*block_size:(block_y+1)*block_size]
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#    ax1.imshow(reconstruction_fbp, cmap = 'gray_r')
    error = reconstruction_fbp - original_image
    error_remove_non_block_parts = reconstruction_shift_radon - original_image

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(14, 8.2))
    ax1.set_title("Original Image")
    ax1.imshow(original_image, cmap='gray')
    ax2.set_title("Sinogram\nZero-Padding Radon")
    ax2.imshow(sinogram_zero_padding, cmap='gray')
    ax3.set_title("Sinogram\nShift Radon")
    ax3.imshow(shifted_sinogram, cmap='gray')
    ax4.set_title("Difference")
    ax4.imshow(sinogram_zero_padding - shifted_sinogram, cmap='gray')

    # iradon of zero-padding radon
    reconstruction_fbp_original = iradon(sinogram_zero_padding, theta=theta, circle=False)
    error_original = reconstruction_fbp_original - original_image

    reconstruction_radon = np.zeros(original_image.shape)
    reconstruction_radon[block_x * block_size:(block_x + 1) * block_size, block_y * block_size:(block_y + 1) * block_size] = reconstruction_fbp_original[block_x * block_size:(block_x + 1) * block_size, block_y * block_size:(block_y + 1) * block_size]
    error_original_remove_non_block_parts = reconstruction_radon - original_image


    # print('error of zero-padding radon: %.8f' % np.sqrt(np.mean((error_original) ** 2)))
    ax5.set_title("Reconstruction\nof Zero-Padding Radon")
    ax5.imshow(reconstruction_fbp_original, cmap='gray')
    ax6.set_title("Reconstruction error\nof Zero-Padding Radon\nMSE:%.8f" % np.sqrt(np.mean((error_original) ** 2)))
    ax6.imshow(error_original, cmap='gray', **imkwargs)

    ax7.set_title("Reconstruction\nof Shift Radon")
    ax7.imshow(reconstruction_fbp, cmap='gray')
    ax8.set_title("Reconstruction error of Shift Radon\nMSE:%.8f" % np.sqrt(np.mean((error) ** 2)))
    ax8.imshow(error, cmap='gray', **imkwargs)



    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def generate_coords(img_shape, block_size):
    assert img_shape[0] == img_shape[1]
    assert img_shape[0] % block_size == 0

    coordinates = []
    for x in range(img_shape[0] // block_size):
        for y in range(img_shape[0] // block_size):
            coordinates.append((x, y))
#            print(coordinates)

    return coordinates

def cal_pad_width(image_shape):
    diagonal = np.sqrt(2) * max(image_shape)
    pad = [int(np.ceil(diagonal - s)) for s in image_shape]
    new_center = [(s + p) // 2 for s, p in zip(image_shape, pad)]
    old_center = [s // 2 for s in image_shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    return pad_width


############################################# Test ##########################################################

@timeit
def test_zero_padding():
    # test settings
    plot = False
    block_size = 20
    image_block = np.ones([block_size, block_size])
    image = np.zeros([160, 160])
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # test image
    image_zero_padding = np.zeros([160, 160])
    image_zero_padding[:block_size, :block_size] = 1.

    # test zero-padding radon transform
    sinogram_zero_padding = zero_padding_radon(image_zero_padding, theta, circle=False, plot=False)
    return sinogram_zero_padding

def test_shift_radon_transform():
    # test settings
    plot = False
    block_size = 20
    image_block = np.ones([block_size, block_size])
    image = np.zeros([160, 160])
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # test shift radon transform
    shifted_sinogram = shift_radon_transform([0,0], image_block, [160, 160], theta, circle=False, plot=False)

    return shifted_sinogram

def test_coordinate_radon(block_coordinate=(0, 0)):
    # test settings
    block_size = 40
    block_x, block_y = block_coordinate
#    image_ori = imread('data/08.png', as_gray=True)
#    image_ori = scipy.io.loadmat('oral_cavity_128_128_16.mat').get('Im_3D')[:,:, 1]
    image_ori = scipy.io.loadmat('D:/Projects/pyProjects/for_us_run_experiments_final/for_us_run_experiments/data/Knee_10.mat').get('img')[:,:, :,0]
    image_ori = np.squeeze(image_ori)
    image_block = image_ori[block_x*block_size:(block_x+1)*block_size, block_y*block_size:(block_y+1)*block_size]

    image = np.zeros(image_ori.shape)
    image[block_x*block_size:(block_x+1)*block_size, block_y*block_size:(block_y+1)*block_size] = image_block   # test image
    theta = np.linspace(0., 180., 180, endpoint=False)


    # test zero-padding radon transform
    sinogram_zero_padding = zero_padding_radon(image, theta, circle=False, plot=False)

    # test shift radon transform
    shifted_sinogram, shifted_sinogram_interp = shift_radon_transform(block_coordinate, image_block, image_ori.shape, theta, circle=False, plot=False)
    scipy.io.savemat('shifted_sinogram_interp.mat', mdict={'img':shifted_sinogram_interp})
    show_reconstructed_img(sinogram_zero_padding, shifted_sinogram_interp, block_coordinate, block_size, image, theta)

@timeit
def time_cost_test_forloop_zero_pdding():
    # test settings
    block_size = 20
    image_block = np.ones([block_size, block_size])
    image = np.zeros([160, 160])
    image[:block_size, :block_size] = 1.  # test image
    num_blocks = int((image.shape[0] / block_size) ** 2)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # for loop
    for _ in range(num_blocks):
        sinogram_zero_padding = zero_padding_radon(image, theta, circle=False, plot=False)

@timeit
def time_cost_test_forloop_shift_radon(block_coordinate=(0, 0)):
    # test settings
    block_size = 20
    block_x, block_y = block_coordinate
    image_block = np.ones([block_size, block_size])
    image = np.zeros([160, 160])
    image[block_x * block_size:(block_x + 1) * block_size, block_y * block_size:(block_y + 1) * block_size] = 1.  # test image
    num_blocks = int((image.shape[0] / block_size) ** 2)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # for loop
    for _ in range(num_blocks):
        shifted_sinogram = shift_radon_transform(block_coordinate, image_block, [160, 160], theta, circle=False, plot=False)

@timeit
def phantom_test_zero_padding():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # zero-padding
    reconstructed_img = np.zeros(image.shape)
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x*block_size: (block_x+1)*block_size, block_y*block_size: (block_y+1)*block_size]
        shifted_sinogram = shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False, plot=False)

        zero_padding_image = np.zeros(image.shape)
        zero_padding_image[block_x*block_size: (block_x+1)*block_size, block_y*block_size: (block_y+1)*block_size] \
            = image[block_x * block_size: (block_x + 1) * block_size, block_y * block_size: (block_y + 1) * block_size]
        sinogram_zero_padding = zero_padding_radon(zero_padding_image, theta, circle=False, plot=False)
        reconstructed_img += iradon(sinogram_zero_padding, theta, circle=False)


@timeit
def phantom_test_shift_radon():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # shift
    reconstructed_img = np.zeros(image.shape)
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]
        shifted_sinogram = shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                 plot=False)
        iradon_img = block_iradon(shifted_sinogram, theta)
        reconstructed_img += iradon_img

    # plt.imshow(reconstructed_img, cmap='gray_r')
    # plt.show()


def phantom_test_block_radon():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # blocks
    t1 = time.time()
    reconstructed_img = np.zeros(image.shape)
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]
        block_sinogram = radon(image_block, theta, circle=False)
        iradon_img = block_iradon(block_sinogram, theta)
        reconstructed_img[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]= iradon_img
    t2=time.time()
    print('running time - shift: %.4f' % (t2-t1))
    plt.imshow(reconstructed_img, cmap='gray_r')
    plt.show()

def phantom_test_block_radon_pad():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    max_coord_x, max_coord_y = max_coordinate(block_size, image.shape)
    border_width = 2
    # blocks
    t1 = time.time()
    reconstructed_img = np.zeros(image.shape)
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate

        if block_x == 0 and block_y == 0:

            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)

            # image_block = image[block_x * block_size:(block_x + 1) * block_size + border_width,
            #                     block_y * block_size:(block_y + 1) * block_size + border_width]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size:(block_x + 1) * block_size + border_width,
            #                     block_y * block_size:(block_y + 1) * block_size + border_width] \
            #               = iradon(block_sinogram, theta, circle=False)

        elif block_x == 0 and block_y == max_coord_y:

            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)

            # image_block = image[block_x * block_size:(block_x + 1) * block_size + border_width,
            #                     block_y * block_size - border_width:(block_y + 1) * block_size]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size:(block_x + 1) * block_size + border_width,
            #                     block_y * block_size - border_width:(block_y + 1) * block_size] \
            #               = iradon(block_sinogram, theta, circle=False)

        elif block_x == max_coord_x and block_y == 0:

            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)

            # image_block = image[block_x * block_size-border_width:(block_x + 1) * block_size,
            #                     block_y * block_size:(block_y + 1) * block_size + border_width]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size,
            #                     block_y * block_size:(block_y + 1) * block_size + border_width] \
            #               = iradon(block_sinogram, theta, circle=False)

        elif block_x == max_coord_x and block_y == max_coord_y:
            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)


            # image_block = image[block_x * block_size-border_width:(block_x + 1) * block_size,
            #                     block_y * block_size-border_width:(block_y + 1) * block_size]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size,
            #                     block_y * block_size-border_width:(block_y + 1) * block_size] \
            #               = iradon(block_sinogram, theta, circle=False)

        elif block_x == 0 and block_y != 0 and block_y < max_coord_y:
            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)


            # image_block = image[block_x * block_size:(block_x + 1) * block_size+border_width,
            #                     block_y * block_size-border_width:(block_y + 1) * block_size+border_width]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size:(block_x + 1) * block_size+border_width,
            #                     block_y * block_size-border_width:(block_y + 1) * block_size+border_width]\
            #               = iradon(block_sinogram, theta, circle=False)

        elif block_x == max_coord_x and block_y != 0 and block_y < max_coord_y:
            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          += iradon(block_sinogram, theta, circle=False)

            # image_block = image[block_x * block_size - border_width:(block_x + 1) * block_size,
            #               block_y * block_size - border_width:(block_y + 1) * block_size + border_width]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size - border_width:(block_x + 1) * block_size,
            #               block_y * block_size - border_width:(block_y + 1) * block_size + border_width] \
            #     = iradon(block_sinogram, theta, circle=False)
        elif block_x != 0 and block_x < max_coord_x and block_y == 0:
            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)


            # image_block = image[block_x * block_size - border_width:(block_x + 1) * block_size + border_width,
            #               block_y * block_size:(block_y + 1) * block_size + border_width]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size - border_width:(block_x + 1) * block_size + border_width,
            #               block_y * block_size:(block_y + 1) * block_size + border_width] \
            #     = iradon(block_sinogram, theta, circle=False)
        elif block_x != 0 and block_x < max_coord_x and block_y == max_coord_y:
            image_block = image[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size]
            block_sinogram = radon(image_block, theta, circle=False)
            reconstructed_img[block_x * block_size:(block_x + 1) * block_size,
                          block_y * block_size:(block_y + 1) * block_size] \
                          = iradon(block_sinogram, theta, circle=False)
            # image_block = image[block_x * block_size - border_width:(block_x + 1) * block_size + border_width,
            #               block_y * block_size - border_width:(block_y + 1) * block_size]
            # block_sinogram = radon(image_block, theta, circle=False)
            #
            # reconstructed_img[block_x * block_size - border_width:(block_x + 1) * block_size + border_width,
            #               block_y * block_size - border_width:(block_y + 1) * block_size] \
            #     = iradon(block_sinogram, theta, circle=False)
        else:
            image_block = image[block_x * block_size-border_width:(block_x + 1) * block_size+border_width,
                                block_y * block_size-border_width:(block_y + 1) * block_size+border_width]
            block_sinogram = radon(image_block, theta, circle=False)

            reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size+border_width,
                                block_y * block_size-border_width:(block_y + 1) * block_size+border_width] \
                          = iradon(block_sinogram, theta, circle=False)

    t2=time.time()
    print('running time - block radon: %.4f' % (t2-t1))
    plt.imshow(reconstructed_img, cmap='gray_r')
    plt.show()


def compare_phantom(theta_times=1, replace_zeros=False):
#    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = imread('data/08.png', as_gray=True)
#    image = scipy.io.loadmat('oral_cavity_128_128_16.mat').get('Im_3D')[:,:, 1]
#    image = rescale(image, scale=0.625, mode='reflect', multichannel=False, anti_aliasing=False)
    num_of_projections = 180
#    data_name = 'Knee_10'
#    data = scipy.io.loadmat('data/{}.mat'.format(data_name), squeeze_me=True)
#    image = np.squeeze(data['img'])[:,:,0]
    theta = np.linspace(0., 180.,num_of_projections*theta_times, endpoint=False)
    block_size = 64
    coordinates = generate_coords(image.shape, block_size)
    # zero-padding
    t1=time.time()

    sinogram_zero_padding = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, num_of_projections*theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        zero_padding_image = np.zeros(image.shape)
        zero_padding_image[block_x * block_size: (block_x + 1) * block_size,
        block_y * block_size: (block_y + 1) * block_size] \
            = image[block_x * block_size: (block_x + 1) * block_size, block_y * block_size: (block_y + 1) * block_size]
        sinogram_zero_padding += zero_padding_radon(zero_padding_image, theta, circle=False, plot=False)
#        sinogram_zero_padding += interp.SART_projection(zero_padding_image, image.shape[0], num_of_projections)
    reconstructed_img_zp = iradon(sinogram_zero_padding, theta, circle=False)
#    reconstructed_img_zp = interp.SART_reconstruction(sinogram_zero_padding, image.shape[0], num_of_projections)
    if replace_zeros:
        zero_index = np.where(image == 0)
        reconstructed_img_zp[zero_index] = 0

    t2=time.time()
    print('running time - zero padding: %.4f' % (t2-t1))
    error_zp = reconstructed_img_zp - image
    print('mse - zero padding %.4f:' % np.sqrt(np.mean((error_zp) ** 2)))

    t3=time.time()
    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, num_of_projections*theta_times])
    shifted_sinogram_interp = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, num_of_projections*theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram1, shifted_sinogram_interp1 = shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                  plot=False)
        shifted_sinogram += shifted_sinogram1
        shifted_sinogram_interp += shifted_sinogram_interp1
#    scipy.io.savemat('shifted_sinogram_interp.mat', mdict={'img' : shifted_sinogram_interp})
    reconstructed_img_s = iradon(shifted_sinogram, theta, circle=False) # Filters available: ramp, shepp-logan, cosine, hamming, hann.
    reconstructed_img_s_interp =  iradon(shifted_sinogram_interp, theta, circle=False)
        
#    reconstructed_img_s = interp.SART_reconstruction(shifted_sinogram, image.shape[0], num_of_projections)
#    reconstructed_img_s_interp = interp.SART_reconstruction(shifted_sinogram_interp, image.shape[0], num_of_projections)
    if replace_zeros:
        reconstructed_img_s[zero_index] = 0
    t4=time.time()
    
    scipy.io.savemat('reconstructed_img_pad_2d.mat', mdict={'img': reconstructed_img_zp})
    scipy.io.savemat('reconstructed_img_2d.mat', mdict={'img': reconstructed_img_s})
    scipy.io.savemat('reconstructed_img_interp_2d.mat', mdict={'img': reconstructed_img_s_interp})
    print('running time - shift: %.4f' % (t4-t3))
    error_s = reconstructed_img_s - image
    print('mse - shift %.4f:' % np.sqrt(np.mean((error_s) ** 2)))
    error_s_interp = reconstructed_img_s_interp - image
    print('mse - shift %.4f:' % np.sqrt(np.mean((error_s_interp) ** 2)))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 10))
    ax1.set_title("Original Image")
    ax1.imshow(image, cmap='gray')
    ax2.set_title("Reconstruction Zero-Padding Radon\nMSE:%.8f" % np.sqrt(np.mean((error_zp) ** 2)))
    ax2.imshow(reconstructed_img_zp, cmap='gray')
    ax3.set_title("Reconstruction Shift Radon\nMSE:%.8f" % np.sqrt(np.mean((error_s) ** 2)))
    ax3.imshow(reconstructed_img_s, cmap='gray')
    ax4.set_title("Reconstruction Shift Radon interp\nMSE:%.8f" % np.sqrt(np.mean((error_s_interp) ** 2)))
    ax4.imshow(reconstructed_img_s_interp, cmap='gray')
    plt.show()

def compare_radon_on_image(image_path, block_size, theta_times=1, replace_zeros=False):
    image = imread(image_path)
    theta = np.linspace(0., 180.,max(image.shape)*theta_times, endpoint=False)
    assert image.shape[0] == image.shape[1]
    assert image.shape[0] % block_size == 0
    coordinates = generate_coords(image.shape, block_size)
    # zero-padding
    t1=time.time()

    sinogram_zero_padding = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]*theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        zero_padding_image = np.zeros(image.shape)
        zero_padding_image[block_x * block_size: (block_x + 1) * block_size,
        block_y * block_size: (block_y + 1) * block_size] \
            = image[block_x * block_size: (block_x + 1) * block_size, block_y * block_size: (block_y + 1) * block_size]
        sinogram_zero_padding += zero_padding_radon(zero_padding_image, theta, circle=False, plot=False)
    reconstructed_img_zp = iradon(sinogram_zero_padding, theta, circle=False)
    if replace_zeros:
        zero_index = np.where(image == 0)
        reconstructed_img_zp[zero_index] = 0

    t2=time.time()
    print('running time - zero padding: %.4f' % (t2-t1))
    error_zp = reconstructed_img_zp - image
    print('mse - zero padding %.4f:' % np.sqrt(np.mean((error_zp) ** 2)))

    t3=time.time()
    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]*theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram += shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                  plot=False)
    reconstructed_img_s = iradon(shifted_sinogram, theta, circle=False) # Filters available: ramp, shepp-logan, cosine, hamming, hann.
    if replace_zeros:
        reconstructed_img_s[zero_index] = 0
    t4=time.time()
    print('running time - shift: %.4f' % (t4-t3))
    error_s = reconstructed_img_s - image
    print('mse - shift %.4f:' % np.sqrt(np.mean((error_s) ** 2)))


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    ax1.set_title("Original Image")
    ax1.imshow(image, cmap='gray')
    ax2.set_title("Reconstruction Zero-Padding Radon\nMSE:%.8f" % np.sqrt(np.mean((error_zp) ** 2)))
    ax2.imshow(reconstructed_img_zp, cmap='gray')
    ax3.set_title("Reconstruction Shift Radon\nMSE:%.8f" % np.sqrt(np.mean((error_s) ** 2)))
    ax3.imshow(reconstructed_img_s, cmap='gray')
    plt.show()


def phantom_test_shift_radon_v2(border_width = 0):
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    max_coord_x, max_coord_y = max_coordinate(block_size, image.shape)
    border_width = border_width

    # shift
    reconstructed_img = np.zeros(image.shape)
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]
        shifted_sinogram = shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                 plot=False)
        iradon_img = block_iradon(shifted_sinogram, theta)
        if block_x == 0 and block_y == 0:

            reconstructed_img[block_x * block_size:(block_x + 1) * block_size + border_width,
                                block_y * block_size:(block_y + 1) * block_size + border_width] \
                             = iradon_img[block_x * block_size:(block_x + 1) * block_size + border_width,
                                        block_y * block_size:(block_y + 1) * block_size + border_width]
        elif block_x == 0 and block_y == max_coord_y:

            reconstructed_img[block_x * block_size:(block_x + 1) * block_size + border_width,
                                block_y * block_size - border_width:(block_y + 1) * block_size] \
                             = iradon_img[block_x * block_size:(block_x + 1) * block_size + border_width,
                                          block_y * block_size - border_width:(block_y + 1) * block_size]

        elif block_x == max_coord_x and block_y == 0:
            reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size,
                                block_y * block_size:(block_y + 1) * block_size + border_width] \
                             = iradon_img[block_x * block_size-border_width:(block_x + 1) * block_size,
                                block_y * block_size:(block_y + 1) * block_size + border_width]
        elif block_x == max_coord_x and block_y == max_coord_y:
            reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size,
                                block_y * block_size-border_width:(block_y + 1) * block_size] \
                             = iradon_img[block_x * block_size-border_width:(block_x + 1) * block_size,
                                block_y * block_size-border_width:(block_y + 1) * block_size]
        else:
            reconstructed_img[block_x * block_size-border_width:(block_x + 1) * block_size+border_width,
                                block_y * block_size-border_width:(block_y + 1) * block_size+border_width] \
                             = iradon_img[block_x * block_size-border_width:(block_x + 1) * block_size+border_width,
                                block_y * block_size-border_width:(block_y + 1) * block_size+border_width]

    plt.imshow(reconstructed_img, cmap='gray_r')
    plt.show()

@timeit
def phantom_test_shift_radon_sum_of_sinogram():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    #zero_index = np.where(image == 0)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2)*image.shape[0])+1, image.shape[1]])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram += shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                    plot=False)
    reconstructed_img = iradon(shifted_sinogram, theta, circle=False, interpolation='linear')
    #reconstructed_img[zero_index] = 0
    plt.imshow(reconstructed_img, cmap='gray_r')
    plt.show()

def test_shift_radon_on_image(image_path, block_size, theta_times=1):
    image = imread(image_path)
    theta = np.linspace(0., 180., max(image.shape) * theta_times, endpoint=False)
    assert image.shape[0] == image.shape[1]
    assert image.shape[0] % block_size == 0
    coordinates = generate_coords(image.shape, block_size)

    t1 = time.time()
    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1] * theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram += shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                  plot=False)
    reconstructed_img_s = iradon(shifted_sinogram, theta,
                                 circle=False)  # Filters available: ramp, shepp-logan, cosine, hamming, hann.

    t2 = time.time()
    print('running time - shift: %.4f' % (t2 - t1))
    error_s = reconstructed_img_s - image
    print('mse - shift %.4f:' % np.sqrt(np.mean((error_s) ** 2)))

    plt.title("Reconstruction Shift Radon\nDiscretized Number of Theta = %d * %d\nMSE:%.8f\nRunning Time: %.4f" %
              (image.shape[0], theta_times, np.sqrt(np.mean((error_s) ** 2)), t2-t1))
    plt.imshow(reconstructed_img_s, cmap='gray')
    plt.show()

@timeit
def phantom_test_shift_radon_theta(theta_times=4):
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    #zero_index = np.where(image == 0)
    theta = np.linspace(0., 180., max(image.shape) * theta_times, endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2)*image.shape[0])+1, image.shape[1]*theta_times])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram += shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                    plot=False)
    reconstructed_img = iradon(shifted_sinogram, theta, circle=False, interpolation='linear')
    #reconstructed_img[zero_index] = 0
    #plt.imshow(reconstructed_img, cmap='gray_r')
    #plt.show()


def compare_phantom_replace_zeros():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False, anti_aliasing=False)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    zero_index = np.where(image == 0)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # zero-padding
    sinogram_zero_padding = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        zero_padding_image = np.zeros(image.shape)
        zero_padding_image[block_x * block_size: (block_x + 1) * block_size,
        block_y * block_size: (block_y + 1) * block_size] \
            = image[block_x * block_size: (block_x + 1) * block_size, block_y * block_size: (block_y + 1) * block_size]
        sinogram_zero_padding += zero_padding_radon(zero_padding_image, theta, circle=False, plot=False)
    reconstructed_img_zp = iradon(sinogram_zero_padding, theta, circle=False)
    reconstructed_img_zp[zero_index] = 0
    error_zp = reconstructed_img_zp - image


    # shift
    shifted_sinogram = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]

        shifted_sinogram += shift_radon_transform(block_coordinate, image_block, image.shape, theta, circle=False,
                                                  plot=False)
    reconstructed_img_s = iradon(shifted_sinogram, theta, circle=False)
    reconstructed_img_s[zero_index] = 0
    error_s = reconstructed_img_s - image

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    ax1.set_title("Original Image")
    ax1.imshow(image, cmap='gray_r')
    ax2.set_title("Reconstruction Zero-Padding Radon\nMSE:%.8f" % np.sqrt(np.mean((error_zp) ** 2)))
    ax2.imshow(reconstructed_img_zp, cmap='gray_r')
    ax3.set_title("Reconstruction Shift Radon\nMSE:%.8f" % np.sqrt(np.mean((error_s) ** 2)))
    ax3.imshow(reconstructed_img_s, cmap='gray_r')
    plt.show()

def shift_radon_transform_upscale(block_coordinate, image_block, original_image_size, theta, upscale_factor=2, circle=False, plot=True):
    """

    :param block_coordinate: coordinate of block. e.g [0, 0] is the upper left block
    :param image_block:
    :param original_image_size:  shape of original image, e.g [160, 160]
    :param theta:
    :param circle:
    :param plot:
    :return:
    """

    block_size = image_block.shape[0]
    sinogram = radon(image_block, theta, circle=circle)
    sinogram = sinogram[::upscale_factor, :]
    # shift the sinogram

    # distance from the left upper corner
    # h: Horizontal v: Vertical
    block_movement_v, block_movement_h = (np.array(block_coordinate) + 1) * block_size - block_size // 2
    H, W = original_image_size
    assert H == W, "Image should be squared."

    T_length = int(np.sqrt(2)*H)+1 # T_length == 227 when image shape is [160, 160]
    shifted_sinogram = np.zeros([T_length, sinogram.shape[1]])
    # distance from center
    block_movement_h_center = H / 2 - block_movement_h
    block_movement_v_center = H / 2 - block_movement_v

    # delta_h = block_movement_h_center
    # delta_v = block_movement_v_center
    delta_v = -block_movement_h_center
    delta_h = block_movement_v_center
    # Quadrant
    # max_coord_x, max_coord_y = max_coordinate(block_size, original_image_size)
    # center_coord_x, center_coord_y = max_coord_x // 2, max_coord_y // 2
    # # print(center_coord_x, center_coord_y)
    # coord_x, coord_y = block_coordinate
    # # # Quadrant 1
    # if coord_x < center_coord_x - 1 and coord_y > center_coord_y:
    #     delta_v = -block_movement_h_center
    #     delta_h = block_movement_v_center
    # # Quadrant 2
    # elif coord_x < center_coord_x - 1 and coord_y < center_coord_y - 1:
    #     delta_v = -block_movement_h_center
    #     delta_h = block_movement_v_center
    #
    # # center parts
    # elif coord_y == center_coord_y - 1 or coord_y == center_coord_y or coord_x == center_coord_x - 1 or coord_x == center_coord_x:
    #     delta_h = block_movement_v_center
    #     delta_v = -block_movement_h_center
    #
    # # Quadrant 3
    # elif coord_x > center_coord_x and coord_y < center_coord_y - 1:
    #     delta_v = -block_movement_h_center
    #     delta_h = block_movement_v_center
    # # Quadrant 4
    # elif coord_x > center_coord_x and coord_y > center_coord_y:
    #     delta_v = -block_movement_h_center
    #     delta_h = block_movement_v_center

    for column_num in range(sinogram.shape[1]):
        # put the sinogram into the same column, but shited rows
        th = (np.pi / 180.0) * theta[column_num]

        row_start = int(round(T_length / 2 - sinogram.shape[0] / 2 + delta_v * np.cos(th) + delta_h * np.sin(th)))
        row_end = row_start + sinogram.shape[0]
        if row_end > 227:
            print(row_end)
            sinogram=sinogram[:-(row_end-227)]

        shifted_sinogram[row_start: row_end, column_num] = sinogram[:, column_num]

    # plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image_block, cmap='gray_r')
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap='gray_r',
                   extent=(0, 160, 0, sinogram.shape[0]), aspect='auto')
        fig.tight_layout()
        plt.show()

    return shifted_sinogram


def phantom_test_coordinate_radon_interpolation_upscale():
    image = imread(data_dir + "/phantom.png", as_gray=True)
    image = rescale(image, scale=0.4, mode='reflect')
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    block_size = 20
    coordinates = generate_coords(image.shape, block_size)
    # zero-padding
    t1 = time.time()

    sinogram_zero_padding = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        zero_padding_image = np.zeros(image.shape)
        zero_padding_image[block_x * block_size: (block_x + 1) * block_size,
        block_y * block_size: (block_y + 1) * block_size] \
            = image[block_x * block_size: (block_x + 1) * block_size, block_y * block_size: (block_y + 1) * block_size]
        sinogram_zero_padding += zero_padding_radon(zero_padding_image, theta, circle=False, plot=False)
    reconstructed_img_zp = iradon(sinogram_zero_padding, theta, circle=False)


    t2 = time.time()
    print('running time - zero padding: %.4f' % (t2 - t1))
    error_zp = reconstructed_img_zp - image
    print('mse - zero padding %.4f:' % np.sqrt(np.mean((error_zp) ** 2)))

    t3 = time.time()
    # shift
    upscale_factor = 2
    shifted_sinogram = np.zeros([int(np.sqrt(2) * image.shape[0]) + 1, image.shape[1]])
    for block_coordinate in coordinates:
        block_x, block_y = block_coordinate
        image_block = image[block_x * block_size: (block_x + 1) * block_size,
                      block_y * block_size: (block_y + 1) * block_size]
        upscaled_image_block = rescale(image_block, upscale_factor)
        shifted_sinogram += shift_radon_transform_upscale(block_coordinate, upscaled_image_block, image.shape, theta, circle=False,
                                                  plot=False)
    reconstructed_img_s = iradon(shifted_sinogram, theta,
                                 circle=False)  # Filters available: ramp, shepp-logan, cosine, hamming, hann.

    t4 = time.time()
    print('running time - shift: %.4f' % (t4 - t3))
    error_s = reconstructed_img_s - image
    print('mse - shift %.4f:' % np.sqrt(np.mean((error_s) ** 2)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    ax1.set_title("Original Image")
    ax1.imshow(image, cmap='gray_r')
    ax2.set_title("Reconstruction Zero-Padding Radon\nMSE:%.8f" % np.sqrt(np.mean((error_zp) ** 2)))
    ax2.imshow(reconstructed_img_zp, cmap='gray_r')
    ax3.set_title("Reconstruction Shift Radon\nMSE:%.8f" % np.sqrt(np.mean((error_s) ** 2)))
    ax3.imshow(reconstructed_img_s, cmap='gray_r')
    plt.show()

if __name__ == '__main__':
    test_coordinate_radon(block_coordinate=[0, 0])
    #lena = mpimg.imread('\data\07.png')
    #x, y = lena.shape
    #fig1 = plt.subplots(1, 1)
    #fig1.imshow(lena, cmap = 'gray_r')
    # test_coordinate_radon(block_coordinate=[2, 3])
    # test_coordinate_radon(block_coordinate=[2, 4])
    # test_coordinate_radon(block_coordinate=[3, 2])
    # test_coordinate_radon(block_coordinate=[4, 2])
    # test_coordinate_radon(block_coordinate=[4, 3])
    # test_coordinate_radon(block_coordinate=[4, 4])
    # test_coordinate_radon(block_coordinate=[4, 5])
    # test_coordinate_radon(block_coordinate=[5, 4])
    # test_coordinate_radon(block_coordinate=[5, 5])
    # test_coordinate_radon(block_coordinate=[5, 2])
    # test_coordinate_radon(block_coordinate=[2, 5])
    # test_coordinate_radon(block_coordinate=[5, 5])
    # test_coordinate_radon(block_coordinate=[5, 1])
    # test_coordinate_radon(block_coordinate=[1, 5])
    # test_coordinate_radon(block_coordinate=[1, 1])

    # time_cost_test_forloop_zero_pdding()
    # time_cost_test_forloop_shift_radon()

    # phantom_test_shift_radon()
    # phantom_test_zero_padding()
    # phantom_test_shift_radon_sum_of_sinogram()
    # phantom_test_shift_radon_theta(theta_times=10)
#
#    compare_phantom(theta_times=1, replace_zeros = False)
    # compare_phantom_large_image()

    # phantom_test_shift_radon_v2(border_width=0)
    # compare_phantom_replace_zeros()

    #compare_radon_on_image(image_path='data/07.png', block_size=32, theta_times=1, replace_zeros=False) # 256 X 256
    # compare_radon_on_image(image_path='data/08.png', block_size=64, theta_times=1, replace_zeros=False) # 512 X 512

    # test_shift_radon_on_image(image_path='data/07.png', block_size=32, theta_times=4)
    # test_shift_radon_on_image(image_path='data/08.png', block_size=64, theta_times=4)

    # phantom_test_block_radon()
    # phantom_test_block_radon_pad()

    # phantom_test_coordinate_radon_interpolation_upscale()
    

#    test full image radon~iradon MSE value
  
#    image = imread('D:\\SpyderProj\\RadonTransform\\data\\08.png', as_gray=True)
#    
#    theta_times=1
#    theta = np.linspace(0., 180.,max(image.shape)*theta_times, endpoint=False)
#    
#    sinogram = radon(image, theta, circle = False)
#    reconstructed_img_zp = iradon(sinogram, theta, circle=False)
#    error= reconstructed_img_zp - image
#    np.sqrt(np.mean((error) ** 2))