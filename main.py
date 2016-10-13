import os
import scipy.misc
import numpy as np

from config import DCGANconfig 
from model import DCGAN

import keras
import tensorflow as tf

def main():
    config = DCGANconfig()
    dcgan = DCGAN(config)
    dcgan.training(data)

