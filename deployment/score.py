import pickle
import json
import numpy 
import time

from sklearn.linear_model import Ridge
from joblib import load

from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector

def init():
    global model

    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name = 'basicmodel')
    model = load(model_path)
    
def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
