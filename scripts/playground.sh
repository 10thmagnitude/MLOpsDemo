$rg="cd-mlops"
$workspace="cdmlops"
$experimentName="diabetes"
$clusterName="cdmlops"
$containerName="diabetesdata"
$fileName="diabetes.csv"
$modelName="basicmodel"
$serviceName="diabetes-aci"
$vmSize="Standard_DS2_V2"
$minNodes=0
$maxNodes=2
$idleSecondsBeforeScaleDown=300

# create a resource group
az group create --name $rg -l westus2 --tags createdBy=colin purpose=demo

# create an ML Workspace
az ml workspace create -g $rg -w $workspace -l westus2 --exist-ok --yes

# create a compute target
az ml computetarget create -n amlcompute -g $rg -w $workspace -n $clusterName -s $vmSize \
    --min-nodes $minNodes --max-nodes $maxNodes \
    --idle-seconds-before-scaledown $idleSecondsBeforeScaleDown --remote-login-port-public-access Disabled

# upload data
$dataStoreName=$(az ml datastore show-default -w $workspace -g $rg --query name -o tsv)
az ml datastore upload -w $workspace -g $rg -n $dataStoreName -p ../data -u $containerName --overwrite true

# create folders for artifacts
mkdir models && mkdir metadata

# train model (basic)
az ml run submit-script -g $rg -w $workspace -e $experimentName --ct $clusterName \
    -d ../training/conda_dependencies.yml --path ../training -c train_basic \
    -t ../metadata/run.json train_model_basic.py $dataStoreName $containerName $fileName

# TODO: invoke python to compare model metrics

# register model
az ml model register -g $rg -w $workspace -n $modelName -f metadata/run.json \
    --asset-path outputs/models/sklearn_diabetes_model.pkl \
    -d "Basic Linear model using diabetes dataset" \
    --model-framework ScikitLearn -t metadata/model.json \
    --tag data=diabetes \
    --tag model=regression \
    --tag type=basic

# download model
$modelId=$(jq -r .modelId metadata/model.json)
az ml model download -g $rg -w $workspace -i $modelId -t ./models --overwrite

# deploy model
# note: deploy command must be in path to config files etc.
cd ../deployment
az ml model deploy -g $rg -w $workspace -n $serviceName -f ../scripts/metadata/model.json \
    --dc aciDeploymentConfig.yml --ic inferenceConfig.yml --overwrite
cd ../scripts