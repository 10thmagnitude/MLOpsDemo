import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from azureml.core.run import Run

# get the Run context
run = Run.get_context()

# get the args
workspaceName = sys.argv[1]
datacontainerName = sys.argv[2]
trainingFileName = sys.argv[3]

# read in the data
fullFileName = './{}/{}/{}'.format(workspaceName, datacontainerName, trainingFileName)
df = pd.read_csv(fullFileName)
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

# train Logistic Regression model
c_value = 0.7
print("c_value: {0:.4f}".format(c_value))
run.log("c_value", c_value)
lr_model = LogisticRegression(C=c_value, random_state=42, solver='liblinear', max_iter=10000)
lr_model.fit(X_train, y_train.ravel())

# training metrics
lr_predict_test = lr_model.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, lr_predict_test)
print("Accuracy: {0:.4f}".format(accuracy_score))
run.log("accuracy", accuracy_score)
print("")

print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, lr_predict_test)
print(confusion_matrix)
print("")

cm_data = {
    "schema_type": "confusion_matrix",
    "schema_version": "v1",
    "data": {
        "class_labels": ["TRUE", "FALSE"],
        "matrix": confusion_matrix.tolist()
    }
}
print(cm_data)
run.log_confusion_matrix("Confusion Matrix JSON", cm_data)
run.log_list("ConfusionMatrix", confusion_matrix.ravel().tolist())

plt.figure(figsize = (10,7))
heatmap = sn.heatmap(confusion_matrix, annot=True, fmt="d")
run.log_image("Confusion Matrix Heat Map", plot=plt)

print("Classification Report:")
print(metrics.classification_report(y_test, lr_predict_test))
print("")

# # train Gaussian Naive Bayes model
# nb_model = GaussianNB()
# nb_model.fit(X_train, y_train.ravel())

# # training metrics
# nb_predict_test = nb_model.predict(X_test)
# accuracy_score = metrics.accuracy_score(y_test, nb_predict_test)
# print("Accuracy: {0:.4f}".format(accuracy_score))
# run.log("accuracy", accuracy_score)
# print("")

# print("Confusion Matrix")
# confusion_matrix = metrics.confusion_matrix(y_test, nb_predict_test)
# print("{0}".format(confusion_matrix))
# run.log_confusion_matrix(name, confusion_matrix)

# print("Classification Report")
# print(metrics.classification_report(y_test, nb_predict_test))
# print("")

# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_diabetes_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(lr_model, model_path)

# upload the model file explicitly into artifacts
print("Copy model into output folder...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Uploaded:")
print(run.get_file_names())

run.complete()