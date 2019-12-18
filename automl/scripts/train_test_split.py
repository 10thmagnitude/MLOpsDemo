import argparse
import os
import azureml.dataprep as dprep
import azureml.core

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_csv(path + '/data', index=False)

print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--input_prepared_data", type=str, help="input data")
parser.add_argument("--input_split_features", type=str, help="input split features")
parser.add_argument("--input_split_labels", type=str, help="input split labels")
parser.add_argument("--output_split_train_x", type=str, help="output split train features")
parser.add_argument("--output_split_train_y", type=str, help="output split train labels")
parser.add_argument("--output_split_test_x", type=str, help="output split test features")
parser.add_argument("--output_split_test_y", type=str, help="output split test labels")

args = parser.parse_args()

print("Argument 1 (input prepared data): %s" % args.input_prepared_data)
print("Argument 2 (input features): %s" % str(args.input_split_features.strip("[]").split("\\;")))
print("Argument 3 (input labels): %s" % str(args.input_split_labels.strip("[]").split("\\;")))
print("Argument 4 (output training features split path): %s" % args.output_split_train_x)
print("Argument 5 (output training labels split path): %s" % args.output_split_train_y)
print("Argument 6 (output test features split path): %s" % args.output_split_test_x)
print("Argument 7 (output test labels split path): %s" % args.output_split_test_y)

input_data = dprep.read_csv(args.input_prepared_data)
output_path = args.output_split_path

split_features = [s.strip().strip("'") for s in args.input_split_features.strip("[]").split("\\;")]
split_labels = [s.strip().strip("'") for s in args.input_split_labels.strip("[]").split("\\;")]

x_df = input_data.keep_columns(split_features).to_pandas_dataframe()
y_df = input_data.keep_columns(split_labels).to_pandas_dataframe()

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

# impute mean for all 0-value readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)

if not (args.output_split_train_x is None and
        args.output_split_test_x is None and
        args.output_split_train_y is None and
        args.output_split_test_y is None):

    write_output(x_train, args.output_split_train_x)
    write_output(y_train, args.output_split_train_y)
    write_output(x_test, args.output_split_test_x)
    write_output(y_test, args.output_split_test_y)
