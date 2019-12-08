import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from azureml.core import Workspace
from azureml.core.experiment import Experiment

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

# get the args
workspace_name = sys.argv[1]
datacontainer_name = sys.argv[2]
training_file_name = sys.argv[3]
subscription_id = sys.argv[4]
resource_group = sys.argv[5]

# get the workspace
ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
experiment = Experiment(workspace=ws, name='automl-diabetes')

# read in the data
full_file_name = './{}/{}/{}'.format(workspace_name, datacontainer_name, training_file_name)
df = pd.read_csv(full_file_name)
print("Columns:", df.columns) 
print("Diabetes data set dimensions : {}".format(df.shape))

# drop uneccessary (duplicate) column - inches for cm
del df['skin']

# map true/false to 1/0
diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# split data for training and testing
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 

# impute mean for all 0-value readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# create the automl config
automl_regressor = AutoMLConfig(
    task='regression',
    experiment_timeout_minutes=15,
    whitelist_models='kNN regressor',
    primary_metric='spearman_correlation', #'r2_score',
    training_data=df,
    label_column_name='diabetes',
    n_cross_validations=5)

run = experiment.submit(automl_regressor, show_output=True)
best_run, fitted_model = run.get_output()
print(best_run)
print(fitted_model)

y_predict = fitted_model.predict(X_test.values)
print(y_predict[:10])

