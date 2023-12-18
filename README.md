# DataMining

Installation Requirements:

python --version
Python 3.12.0 

imports:
import os
import math
import random
import sys
import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, and MinMaxScaler 
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
