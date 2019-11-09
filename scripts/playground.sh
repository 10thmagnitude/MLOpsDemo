#!/bin/bash
set -e

rg="cd-mlops"
location="westus2"
workspace="cdmlops"
experimentName="diabetes"
clusterName="cdmlops"
containerName="diabetesdata"
fileName="diabetes_pima.csv"
trainingScript="train_pima_model.py"
modelName="basicmodel"
serviceName="diabetes-aci"
vmSize="Standard_DS2_V2"
minNodes=0
maxNodes=2
idleSecondsBeforeScaleDown=300

# assumes `az login` has already happened

# create a resource group
# echo "Creating resource group $rg in $location"
# az group create --name $rg -l $location --tags createdBy=colin purpose=demo

# # create an ML Workspace
# echo "Creating ML workspace $workspace"
# az ml workspace create -g $rg -w $workspace -l $location --exist-ok --yes

# # create a compute target
# echo "Creating compute target $clusterName"
# az ml computetarget create amlcompute -g $rg -w $workspace -n $clusterName -s $vmSize \
#     --min-nodes $minNodes --max-nodes $maxNodes \
#     --idle-seconds-before-scaledown $idleSecondsBeforeScaleDown --remote-login-port-public-access Disabled

# # upload data
# echo "Uploading training data"
# dataStoreName=$(az ml datastore show-default -w $workspace -g $rg --query name -o tsv)
# az ml datastore upload -w $workspace -g $rg -n $dataStoreName -p ../data -u $containerName --overwrite true

# # create folders for artifacts
# mkdir models -p && mkdir metadata -p

# # train model (basic)
# rm -f metadata/run.json
# echo "Training model using experiment $experimentName and script $trainingScript"
# az ml run submit-script -g $rg -w $workspace -e $experimentName --ct $clusterName \
#     -d ../training/conda_dependencies.yml --path ../training -c train_basic \
#     -t metadata/run.json $trainingScript $dataStoreName $containerName $fileName

# TODO: invoke python to compare model metrics

# register model
# echo "Registering model $modelName"
# az ml model register -g $rg -w $workspace -n $modelName -f metadata/run.json \
#     --asset-path outputs/models/sklearn_diabetes_model.pkl \
#     -d "Basic Linear model using diabetes dataset" \
#     --model-framework ScikitLearn -t metadata/model.json \
#     --tag data=diabetes \
#     --tag model=regression \
#     --tag type=basic

# download model
modelId=$(jq -r .modelId metadata/model.json)
az ml model download -g $rg -w $workspace -i $modelId -t ./models --overwrite

# deploy model
# note: deploy command must be in path to config files etc.
echo "Deploying model to service $serviceName"
cd ../deployment
az ml model deploy -g $rg -w $workspace -n $serviceName -f ../scripts/metadata/model.json \
    --dc aciDeploymentConfig.yml --ic inferenceConfig.yml --overwrite
cd ../scripts