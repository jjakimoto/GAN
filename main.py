import os
import scipy.misc
import numpy as np
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# from config import DCGANconfig 
from conig import mnistConfig
from model import DCGAN
from utils import get_images

def main(data_path):
    # config = DCGANConfig()
    config = mnistConfig()
    dcgan = DCGAN(config)
    # data = get_images(data_path)
    data = mnist.train.images.reshape([-1, 28, 28])
    dcgan.train(data)

if __name__ = '__main__':
    main()
