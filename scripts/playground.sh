$rg="cd-mlops"
$workspace="cdmlops"
$experimentName="diabetes"
$clusterName="cdmlops"
$containerName="diabetesdata"
$fileName="diabetes.csv"
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

# register model
az ml register -g $rg -w $workspace -n $modelName -f metadata/run.json \
    --asset-path outputs/models/sklearn_diabetes_model.pkl \
    -d "Basic Linear model using diabetes dataset"
    --tag "data"="diabetes" \
    --tag "model"="regression" \
    --tag "type"="basic" \
    --model-framework ScikitLearn \
    -t metadata/model.json