import pandas as pd
import sys
import os
import logging

from azureml.core import Workspace, Dataset, Datastore
from azureml.core.experiment import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.train.automl.runtime import AutoMLStep

from azureml.data.data_reference import DataReference 
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep

from azureml.pipeline.core import Pipeline

workspace_name = 'cdmlops'
compute_target_name = 'cdmlops'
dataset_name = 'diabetesdata/diabetes_pima.csv'
subscription_id = '7cb97533-0a52-4037-a51e-8b8d707367ad'
resource_group = 'cd-mlops'

# get the args
# workspace_name = sys.argv[1]
# compute_target_name = sys.argv[2]
# dataset_name = sys.argv[3]
# subscription_id = sys.argv[4]
# resource_group = sys.argv[5]

# get the workspace
ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
experiment = Experiment(workspace=ws, name='automl-diabetes')
aml_compute = AmlCompute(ws, compute_target_name)

# read in the data
datastore = ws.get_default_datastore()
blob_diabetes_data = DataReference(
    datastore=datastore,
    data_reference_name="diabetes_data",
    path_on_datastore="diabetes_data/diabetes_pima.csv")

# Create a new runconfig object
aml_run_config = RunConfiguration()
aml_run_config.target = aml_compute
aml_run_config.environment.docker.enabled = True
aml_run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base:0.2.1"
aml_run_config.environment.python.user_managed_dependencies = False
aml_run_config.auto_prepare_environment = True
aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['pandas', 'scikit-learn'], 
    pip_packages=['azureml-sdk', 'azureml-dataprep', 'azureml-train-automl'], 
    pin_sdk_version=False)

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

feature_names = str(['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']).replace(",", ";")
label_names = str(['diabetes']).replace(",", ";")

output_split_train_x = PipelineData("diabetes_automl_split_train_x", datastore=datastore)
output_split_train_y = PipelineData("diabetes_automl_split_train_y", datastore=datastore)
output_split_test_x = PipelineData("diabetes_automl_split_test_x", datastore=datastore)
output_split_test_y = PipelineData("diabetes_automl_split_test_y", datastore=datastore)

train_test_split_step = PythonScriptStep(
    name="Split train and test data",
    script_name="train_test_split.py", 
    arguments=["--input_prepared_data", prepared_data, 
               "--input_split_features", feature_names,
               "--input_split_labels", label_names,
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

automl_settings = {
    "name": "AutoML_Diabetes_Experiment",
    "iteration_timeout_minutes": 15,
    "iterations": 25,
    "n_cross_validations": 5,
    "primary_metric": 'spearman_correlation',  # 'r2_score'
    "preprocess": False,
    "max_concurrent_iterations": 8,
    "verbosity": logging.INFO
}

automl_config = AutoMLConfig(task='regression',
                             debug_log = 'auto_ml_errors.log',
                             compute_target=aml_compute,
                             path=os.getcwd() + '/automl',
                             data_script='get_data.py',
                             **automl_settings,
                            )

train_step = AutoMLStep(
    name='AutoML_Regression',
    automl_config=automl_config,
    inputs=[output_split_train_x, output_split_train_y],
    allow_reuse=True,
    hash_paths=[os.path.realpath(scripts_folder)])

pipeline_steps = [train_step]

pipeline = Pipeline(workspace = ws, steps=pipeline_steps)
print("Pipeline is built.")

pipeline_run = experiment.submit(pipeline, regenerate_outputs=False)

print("Pipeline submitted for execution.")
pipeline_run.wait_for_completion()