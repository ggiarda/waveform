import csv

import numpy as np
import matplotlib.pyplot as plt

import pysr
from pysr import PySRRegressor

from sympy import Piecewise

#--------------------------------------Data Reading----------------------------------------------

# Specify the CSV file path
csv_file_path = 'time_data.csv'

# Read from CSV and create NumPy arrays
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=None)

# Extract columns

X = data[:, 0]
y = data[:, 1]
q = data[:, 2]

y = y.reshape(-1, 1)

# Use this if you don't want to use q as a feature
        
X = X.reshape(-1, 1)

# Use this if you want to use q as a feature

#X = np.column_stack((X, q))

#---------------------------------------------------Fitting---------------------------------------------------------------

model_time = PySRRegressor(
    model_selection="accuracy",
    populations = 24,
    niterations= 200,
    binary_operators=["+", "*", "-", "/", 
                      "^",
                      #"cond(x, y) = x < 0 ? Float32(0) : y",
                     ],
    unary_operators=[
        #"square",
        #"cube",
        #"sqrt",
        "exp",
        "sin",
        "cos",
        #"tan",
        #"log",
        #"sinh",
        #"cosh",
        #"tanh",
        #"atan",
        #"asinh",
        #"acosh",
    ],
    
    loss="L2DistLoss()",
    
    maxsize = 45, # Default is 20
    
    #batching=True, # batching is recommended for dataset with > 10,000 datapoints
    #batch_size = 1000, # Default is 50
    
    extra_sympy_mappings={
                          #"cond": lambda x, y: Piecewise((0.0, x < 0), (y, True)), 
                         },
    
    constraints = {
                   "^": (9, 1)
                   #"cond": (8, -1)
                  },
    nested_constraints = {
                          "sin": {"sin": 0, "cos": 0}, 
                          "cos": {"sin": 0, "cos": 0},
                          #"square":{"square": 0,},
                          #"cond":{"cond": 0},
                         }, 
    
    weight_optimize = 0.001,
)

# Use this if you don't want to use q as a feature
        
model_time.fit(X, y,variable_names = ["t"])

# Use this if you want to use q as a feature

#model_freq.fit(X, y,variable_names = ["t","q"])
