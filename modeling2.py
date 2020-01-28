# %%
import os
import pickle
import time
from os.path import dirname, abspath

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (BatchNormalization, ConvLSTM2D, Dense,
                                     Input, LeakyReLU, Conv3D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from_generator = tf.data.Dataset.from_generator
ROOT_DIR = dirname(abspath(__file__))
import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from DataPrep import DataPrepDlib
