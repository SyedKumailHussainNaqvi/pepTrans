# imports.py

import os
import math
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import RootMeanSquaredError
import timeit as ti
import sys
from types import SimpleNamespace 

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score, 
    matthews_corrcoef,
    average_precision_score,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    concatenate,
    Conv1D,
    MaxPooling1D,
    Flatten,
    GlobalAveragePooling1D,
    Bidirectional,
    LSTM,
    Add,
    Masking
)
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow_addons.optimizers import AdamW

from keras.layers import (
    Lambda,
    Reshape,
    GlobalMaxPooling1D,
)
from keras.models import Model as KerasModel
from keras.regularizers import l2 as keras_l2
from keras.initializers import he_normal as keras_he_normal

from tensorflow.python.keras import backend as K