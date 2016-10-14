import numpy as np
from scipy.misc import imresize
from PIL import Image
from os import listdir
from os.path import join, isfile, isdir


def get_images(image_dir='/path/to/your/images'):
    """get images from your local path"""
    images = []
    files = listdir(image_dir)
    for f in iter(files):
        file_path = join(image_dir, f)
        if isfile(file_path):
            images.append(np.array(Image.open(file_path)))
        elif isdir(file_path):
            images += get_images(file_path)
        else:
            pass
    return images

def combine_images(images):
    """combine list of image data
    Args:
        images(list): each element has 2 or 3 dimention
    Return:
        combined image
    """
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    # shape of images 
    # generated_images.shape == (n_images, height, width, c_dim)
    shape = images.shape[1:]
    combined_image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     images.dtype)
    for index, img in enumerate(images):
        i = int(index/width)
        j = index % width
        combined_image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = img
    return image

def resize_data(data, width, height, color_dim=None, is_color=True):
    """resize data for trainining dcgan
    Args:
        data: list of image data, each of which has a shape,
            (width, height, color_dim) if is_color==True
            (width, height) otherwisei
    """
    if is_color:
        converted_data = np.array([imresize(d, [width, height]) for d in data
                                if (len(d.shape)==3 and d.shape[-1] == c_dim)])
        # transform from [0, 255] to [-1, 1]
        converted_data = (converted_data.astype(np.float32) - 127.5)/127.5
    else:
        # gray scale data
        converted_data = np.array([imresize(d, [width, height]) for d in data
                                if (len(d.shape)==2)])
        # transform from [0, 1] to [-1, 1]
        converted_data = (converted_data.astype(np.float32) - 0.5)/0.5
        converted_data = converted_data.reshape(converted_data.shape + (1,))

    return converted_data
