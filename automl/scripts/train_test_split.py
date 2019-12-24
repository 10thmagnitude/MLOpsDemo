import argparse
import os
import azureml.dataprep as dprep
import azureml.core
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_csv(path + '/data', index=False)

print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--input_prepared_data", type=str, help="input data")
parser.add_argument("--output_split_train_x", type=str, help="output split train features")
parser.add_argument("--output_split_train_y", type=str, help="output split train labels")
parser.add_argument("--output_split_test_x", type=str, help="output split test features")
parser.add_argument("--output_split_test_y", type=str, help="output split test labels")

args, unknown = parser.parse_known_args()
if (unknown):
  print("Unknown args:")
  print(unknown)

print("Argument 1 (input prepared data): %s" % args.input_prepared_data)
print("Argument 2 (output training features split path): %s" % args.output_split_train_x)
print("Argument 3 (output training labels split path): %s" % args.output_split_train_y)
print("Argument 4 (output test features split path): %s" % args.output_split_test_x)
print("Argument 5 (output test labels split path): %s" % args.output_split_test_y)

input_data = dprep.read_csv(args.input_prepared_data)

feature_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
label_names = ['diabetes']

print("Features:")
print(feature_names)
print("Labels:")
print(label_names)

x_df = input_data.keep_columns(feature_names).to_pandas_dataframe()
y_df = input_data.keep_columns(label_names).to_pandas_dataframe()

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

# impute mean for all 0-value readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
x_train_imputed = pd.DataFrame(fill_0.fit_transform(x_train), columns=x_df.columns)
x_test_imputed = pd.DataFrame(fill_0.fit_transform(x_test), columns=x_df.columns)

if not (args.output_split_train_x is None and
        args.output_split_test_x is None and
        args.output_split_train_y is None and
        args.output_split_test_y is None):

    write_output(x_train_imputed, args.output_split_train_x)
    write_output(x_test_imputed, args.output_split_test_x)
    write_output(y_train, args.output_split_train_y)
    write_output(y_test, args.output_split_test_y)
