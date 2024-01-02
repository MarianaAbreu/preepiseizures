# Implementation of Seizure Prediction by modules
#
# 1. Input Data
# 2. Train Test Devision
# 3. Segmentation - if necessary 
# 4. Extraction - if necessary
# 5. Normalization 
# 6. Augmentation
# 4. Model compilation
# 5. Model training
# 6. Model evaluation
#
# Created by: Mariana Abreu
# Created on: 23/11/2023

# built-in libraries
import os

# third-party libraries
import numpy as np
import pandas as pd

# local libraries
from preepiseizures.src import Patient


def load_data(path):
    """
    Load data from path
    :param path: path to data
    :return: data
    """
    data = pd.read_csv(path, header=None)
    return data


if __name__ == "__main__":
    patient = 'BLIW'
    # 1. Input Data

    # 2. Train Test Devision
    # 3. Segmentation - if necessary 
    # 4. Extraction - if necessary
    # 5. Normalization 
    # 6. Augmentation
    # 4. Model compilation
    # 5. Model training
    # 6. Model evaluation
    pass