import pandas as pd
import sys
import os
import logging
import argparse

from azureml.core import Workspace, Dataset, Datastore
from azureml.core.experiment import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.train.automl.runtime import AutoMLStep

from azureml.data.data_reference import DataReference 
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
import azureml.dataprep as dprep

from sklearn import metrics

# parse the args
parser = argparse.ArgumentParser("automl_model")
parser.add_argument("--workspace", type=str, help="name of workspace", default='cdmlops')
parser.add_argument("--compute_target", type=str, help="name of compute target", default='cdmlops')
parser.add_argument("--dataset", type=str, help="name of dataset", default='diabetesdata/diabetes_pima.csv')
parser.add_argument("--subscription_id", type=str, help="subscription id", default='7cb97533-0a52-4037-a51e-8b8d707367ad')
parser.add_argument("--resource_group", type=str, help="name of resource group", default='cd-mlops')
parser.add_argument("--build_number", type=str, help="build number tag value", default='1.0.0')

args = parser.parse_args()
workspace_name = args.workspace
compute_target_name = args.compute_target
dataset_name = args.dataset
subscription_id = args.subscription_id
resource_group = args.resource_group
build_number = args.build_number

# workspace_name = 'cdmlops'
# compute_target_name = 'cdmlops'
# dataset_name = 'diabetesdata/diabetes_pima.csv'
# subscription_id = '7cb97533-0a52-4037-a51e-8b8d707367ad'
# resource_group = 'cd-mlops'
# build_number = '1.0.0'

print("workspace_name: %s" % workspace_name)
print("compute_target_name: %s" % compute_target_name)
print("dataset_name: %s" % dataset_name)
print("subscription_id: %s" % subscription_id)
print("resource_group: %s" % resource_group)

# get the workspace
print("Getting a reference to workspace %s" % workspace_name)
ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
experiment = Experiment(workspace=ws, name='automl-diabetes')
aml_compute = AmlCompute(ws, compute_target_name)

# read in the data
print("Getting a reference to default datastore")
datastore = ws.get_default_datastore()

print("Preparing the 'prep data' step")
blob_diabetes_data = DataReference(
    datastore=datastore,
    data_reference_name="diabetes_data",
    path_on_datastore="diabetesdata/diabetes_pima.csv")

# Create a new runconfig object
aml_run_config = RunConfiguration()
aml_run_config.target = aml_compute
aml_run_config.environment.docker.enabled = True
aml_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE  # "mcr.microsoft.com/azureml/base:intelmpi2018.3-ubuntu16.04"
aml_run_config.environment.python.user_managed_dependencies = False
aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
    # conda_packages=['numpy==1.18.0', 'pandas', 'scikit-learn'], 
    # pip_packages=['azureml-sdk', 'azureml-core', 'azureml-dataprep', 'azureml-dataprep[pandas]', 'azureml-train-automl']
    conda_packages=['pandas', 'scikit-learn','numpy==1.18.0'], 
    pip_packages=['azureml-sdk', 'azureml-dataprep', 'azureml-train-automl'], 
    pin_sdk_version=False
)

scripts_folder = './scripts'
prepared_data = PipelineData("diabetes_data_prep", datastore=datastore)

prep_data_step = PythonScriptStep(
    name="Prep diabetes data",
    script_name="prep_data.py", 
    arguments=["--input_file", blob_diabetes_data, 
               "--output_path", prepared_data],
    inputs=[blob_diabetes_data],
    outputs=[prepared_data],
    compute_target=aml_compute,
    runconfig=aml_run_config,
    source_directory=scripts_folder,
    allow_reuse=True
)

print("Preparing the 'split train and data' step")
output_split_train_x = PipelineData("diabetes_automl_split_train_x", datastore=datastore)
output_split_train_y = PipelineData("diabetes_automl_split_train_y", datastore=datastore)
output_split_test_x = PipelineData("diabetes_automl_split_test_x", datastore=datastore)
output_split_test_y = PipelineData("diabetes_automl_split_test_y", datastore=datastore)

train_test_split_step = PythonScriptStep(
    name="Split train and test data",
    script_name="train_test_split.py", 
    arguments=["--input_prepared_data", prepared_data, 
               "--output_split_train_x", output_split_train_x,
               "--output_split_train_y", output_split_train_y,
               "--output_split_test_x", output_split_test_x,
               "--output_split_test_y", output_split_test_y],
    inputs=[prepared_data],
    outputs=[output_split_train_x, output_split_train_y, output_split_test_x, output_split_test_y],
    compute_target=aml_compute,
    runconfig=aml_run_config,
    source_directory=scripts_folder,
    allow_reuse=True
)

print("Preparing the 'autoML' step")
automl_settings = {
    "name": "AutoML_Diabetes_Experiment",
    "iteration_timeout_minutes": 15,
    "iterations": 25,
    "n_cross_validations": 5,
    "primary_metric": 'average_precision_score_weighted', # regression metrics: 'spearman_correlation',  # 'r2_score'
    "preprocess": False,
    "num_classes": 2,
    "max_concurrent_iterations": 8,
    "verbosity": logging.INFO
}

automl_config = AutoMLConfig(task='classification', # regression
                             debug_log = 'auto_ml_errors.log',
                             compute_target=aml_compute,
                             path=os.path.realpath(scripts_folder),
                             data_script='get_data.py',
                             **automl_settings
                            )

train_step = AutoMLStep(
    name='AutoML_Classification',
    automl_config=automl_config,
    inputs=[output_split_train_x, output_split_train_y],
    allow_reuse=True)

print("Building pipeline")
pipeline_steps = [train_step]
pipeline = Pipeline(workspace = ws, steps=pipeline_steps)

print("Submitting pipeline")
pipeline_run = experiment.submit(pipeline, regenerate_outputs=False)

print("Waiting for pipeline completion")
pipeline_run.wait_for_completion()

def get_download_path(download_path, output_name):
    output_folder = os.listdir(download_path + '/azureml')[0]
    path =  download_path + '/azureml/' + output_folder + '/' + output_name
    return path

def fetch_df(step, output_name):
    output_data = step.get_output_data(output_name)
    
    download_path = './outputs/' + output_name
    output_data.download(download_path)
    df_path = get_download_path(download_path, output_name) + '/data'
    return dprep.auto_read_file(path=df_path)

print("Get the best model")
# workaround to get the automl run as its the last step in the pipeline 
# and get_steps() returns the steps from latest to first
for step in pipeline_run.get_steps():
    automl_step_run_id = step.id
    print(step.name)
    print(automl_step_run_id)
    break

automl_run = AutoMLRun(experiment = experiment, run_id=automl_step_run_id)
best_run, fitted_model = automl_run.get_output()
print(best_run)
print(fitted_model)

print("Get metrics")
children = list(automl_run.get_children())
metricslist = {}
for run in children:
    properties = run.get_properties()
    metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
    metricslist[int(properties['iteration'])] = metrics

rundata = pd.DataFrame(metricslist).sort_index(1)
print("Rundata:")
print(rundata)

print("Get the test data")
split_step = pipeline_run.find_step_run(train_test_split_step.name)[0]

# x_train = fetch_df(split_step, output_split_train_x.name).to_pandas_dataframe()
# y_train = fetch_df(split_step, output_split_train_y.name).to_pandas_dataframe()

x_test = fetch_df(split_step, output_split_test_x.name).to_pandas_dataframe()
y_test = fetch_df(split_step, output_split_test_y.name).to_pandas_dataframe()

print("Test the model")
print(x_test)
y_predict = fitted_model.predict(x_test.values)
y_actual = y_test.iloc[:,0].values.tolist()

print("Prediction results:")
prediction_results = pd.DataFrame({'Actual':y_actual, 'Predicted':y_predict}).head(10)
print(prediction_results)

print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
print(confusion_matrix)

print("Model pkl file is model.pkl in %s" % os.getcwd())

# set the best_run.id variable for azure pipelines
print("##vso[task.setvariable variable=bestRunId;]{}".format(best_run.id.replace("'", '')))