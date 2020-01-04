import pickle
import json
import numpy 
import time
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from joblib import load

from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    global model

    print ("model initialized at " + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name = 'diabetesmodel')
    model = load(model_path)

input_sample = pd.DataFrame(data=[{
    "num_preg": 1,
    "glucose_conc": 1,
    "diastolic_bp": 1,
    "thickness": 1,
    "insulin": 1,
    "bmi": 2.2,
    "diab_pred": 2.2,
    "age": 1
}])
output_sample = np.array([0])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        print ("Invoking model for data")
        result = model.predict(data)
        res_string = json.dumps({"result": result.tolist()})
        
        print ("Results:")
        print (res_string)
        return res_string
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
