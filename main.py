import os
import scipy.misc
import numpy as np
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# from config import DCGANconfig 
from config import mnistConfig
from model import DCGAN
from utils import get_images

def main(data_path):
    # config = DCGANConfig()
    config = mnistConfig()
    dcgan = DCGAN(config)
    # data = get_images(data_path)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data = mnist.train.images.reshape([-1, 28, 28])
    dcgan.train(data)

if __name__ == '__main__':
    main()
