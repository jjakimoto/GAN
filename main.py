import os
import scipy.misc
import numpy as np

from config import DCGANconfig 
from model import DCGAN
from utils import get_data

import keras
import tensorflow as tf

def main(data_path):
    config = DCGANconfig()
    dcgan = DCGAN(configo)
    data = get_data(data_path)
    dcgan.train(data)


