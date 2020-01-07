# MLOps Demo

This repo shows how to use Azure Machine Learning (ML) Services and Azure DevOps to manage machine learning using DevOps principles and practices.

## Model
A model is trained on Pima Indian Diabetes data to predict the likelihood of diabetes from health markers. The training is done using Azure Machine learning workspaces and every step of the process is in code. The trained model is deployed to Azure Container Instances (ACI) for consuming.

## Website
A simple website is deployed that consumes a deployed model. See [this page](https://cd-diabetes-dev.azurewebsites.net/diabetes) for a demo.

## Status

### Model Pipeline

Stage | Status
---|---
Provision Workspace|[![Build Status](https://10m.visualstudio.com/Demos/_apis/build/status/mlops/mlops.train-model?branchName=master&stageName=Provision%20Workspace)](https://10m.visualstudio.com/Demos/_build/latest?definitionId=79&branchName=master)
Train Model|[![Build Status](https://10m.visualstudio.com/Demos/_apis/build/status/mlops/mlops.train-model?branchName=master&stageName=Train%20Model)](https://10m.visualstudio.com/Demos/_build/latest?definitionId=79&branchName=master)
Deploy to ACI|[![Build Status](https://10m.visualstudio.com/Demos/_apis/build/status/mlops/mlops.train-model?branchName=master&stageName=Deploy%20Model%20to%20DEV)](https://10m.visualstudio.com/Demos/_build/latest?definitionId=79&branchName=master)

### Site Pipeline
Stage | Status
---|---
Build Site|[![Build Status](https://10m.visualstudio.com/Demos/_apis/build/status/mlops/mlops.webapp?branchName=master&stageName=Build%20website)](https://10m.visualstudio.com/Demos/_build/latest?definitionId=81&branchName=master)
Provision Infra and Deploy to DEV|[![Build Status](https://10m.visualstudio.com/Demos/_apis/build/status/mlops/mlops.webapp?branchName=master&stageName=Provision%20Infrastructure)](https://10m.visualstudio.com/Demos/_build/latest?definitionId=81&branchName=master)

## URLs
1. Scoring URL: Get this from the ML Portal under Endpoints.
2. For the Swagger, change `/score` to `/swagger.json` in the URL to get the OpenAPI definition.