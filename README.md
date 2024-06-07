# DataMining

The process of automatic discovery of patterns and knowledge from large
data repositories, including databases, data warehouses, Web, document
collections, and data streams. The basic topics of data mining,
including data preprocessing, frequent pattern and association rule mining,
correlation analysis, classification and prediction, and clustering, as well as
advanced topics covering the techniques and applications of data mining in Web,
text, big data, social networks, and computational journalism.

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
