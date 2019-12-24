import argparse
import os
import azureml.dataprep as dprep
import azureml.core

print("Prepare data for training")

parser = argparse.ArgumentParser("prep_data")
parser.add_argument("--input_file", type=str, help="input raw data file")
parser.add_argument("--output_path", type=str, help="output prepped data path")

args, unknown = parser.parse_args()

print("Argument 1 (input training data file): %s" % args.input_file)
print("Argument 2 (output prepped training data path) %s" % args.output_path)

input_file = dprep.read_csv(args.input_file)

prepped_data = (input_file
                .drop_columns(columns='skin')  # skin is same as thickness with another unit (inches/cm)
                .replace(columns='diabetes', find="TRUE", replace_with="1")
                .replace(columns='diabetes', find="FALSE", replace_with="0")
                .set_column_types(type_conversions={
                    'diabetes': dprep.TypeConverter(data_type=dprep.FieldType.INTEGER) }))

if not (args.output_path is None):
    os.makedirs(args.output_path, exist_ok=True)
    print("%s created" % args.output_path)
    write_df = prepped_data.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_path))
    write_df.run_local()