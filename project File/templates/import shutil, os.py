import shutil, os
from zipfile import ZipFile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import RMSprop, Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, Xception, InceptionV3, EfficientNetB0, MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model


from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')