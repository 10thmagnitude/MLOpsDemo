$rg="cd-mlops"
$workspace="cdmlops"
$clusterName="cdmlops"
$vmSize="Standard_DS2_V2"
$minNodes=0
$maxNodes=2
$idleSecondsBeforeScaleDown=300

# create a resource group
az group create --name $rg -l westus2 --tags createdBy=colin purpose=demo

# create an ML Workspace
az ml workspace create -g $rg -w $workspace -l westus2 --exist-ok --yes

# create a compute target
az ml computetarget create mlopscompute -g $rg -w $workspace -n $clusterName -s $vmSize \
    --min-nodes $minNodes --max-nodes $maxNodes --idle-seconds-before-scaledow $idleSecondsBeforeScaleDown

# upload data
$dataStoreName=$(az ml datastore show-default -w $workspace -g $rg --query name -o tsv)
az ml datastore upload -w $workspace -g $rg -n $dataStoreName -p data -u diabetes --overwrite true