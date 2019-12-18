import os

from azureml.core import Dataset, Run

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def get_data():
    run = Run.get_context()

    print("OS===")
    for k, v in os.environ.items():
        print(f'{k}={v}')

    print("run obj===")
    print(run)

    print("run obj attrs===")
    for attr in dir(run):
        print("obj.%s = %r" % (attr, getattr(run, attr)))

    # get the input dataset by name
    ds = run.input_datasets['diabetes_dataset']

    # get a pandas dataframe
    df = ds.to_pandas_dataframe()
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

    return {
        'X': X_train,
        'y': y_train,
        'X_test': X_test,
        'y_test': y_test
    }