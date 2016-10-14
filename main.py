import os
import scipy.misc
import numpy as np

from config import DCGANconfig 
from model import DCGAN
from utils import get_images

import keras
import tensorflow as tf

def main(data_path):
    # config = DCGANconfig()
    config = mnistConfig()
    dcgan = DCGAN(config)
    data = get_data(data_path)
    dcgan.train(data)


