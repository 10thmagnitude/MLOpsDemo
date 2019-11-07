import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core.run import Run
from azureml.core.model import Model

# get the args
workspaceName = sys.argv[1]
datacontainerName = sys.argv[2]
trainingFileName = sys.argv[3]

fullFileName = './{}/{}/{}'.format(workspaceName, datacontainerName, trainingFileName)
run = Run.get_context()

print("Loading training data from {}...".format(fullFileName))
diabetes = pd.read_csv(fullFileName)
print("Columns:", diabetes.columns) 
print("Diabetes data set dimensions : {}".format(diabetes.shape))

y = diabetes.pop('Y')
X_train, X_test, y_train, y_test = train_test_split(diabetes, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Training the model...")
# Randomly pick alpha
alphas = np.arange(0.0, 1.0, 0.05)
alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
print("alpha:", alpha)
run.log("alpha", alpha)
reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])
run.log_list("coefficients", reg.coef_)

print("Evaluate the model...")
preds = reg.predict(data["test"]["X"])
mse = mean_squared_error(preds, data["test"]["y"])
print("Mean Squared Error:", mse)
run.log("mse", mse)

# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_diabetes_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(reg, model_path)

# upload the model file explicitly into artifacts
print("Copy model into output folder...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Uploaded:")
print(run.get_file_names())

run.complete()