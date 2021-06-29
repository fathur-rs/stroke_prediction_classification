import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from terminaltables import SingleTable
import os
import time

# ergid
print('=== TRAINING.PY ==='), time.sleep(2)
def data():
    url = 'https://raw.githubusercontent.com/fathur-rs/uas/master/healthcare-dataset-stroke-data.csv'
    data = pd.read_csv(url)
    print('Reading Dataset...')
    return data_clean(data)

# melodi
def data_clean(data):
    df = data.loc[data["gender"] != 'Other']
    df.dropna(axis=0, inplace=True)
    return data_prep(df)

#fathur
def data_prep(df):
    label_encoder = preprocessing.LabelEncoder()
    obj_df = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for i in obj_df:
        df[i] = label_encoder.fit_transform(df[i])
    print('Data Preprocessing...')
    return splitting_data(df)

#anisyaul
def splitting_data(df):
    X = df.drop(['id', 'stroke'], axis=1)
    Y = df.stroke
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.29, random_state=42)
    print('Split the data...')
    return resampling(X_train, X_test, y_train, y_test)

#razan
def resampling(X_train, X_test, y_train, y_test):
    print('Resampling Train Data...')
    smote = SMOTE(sampling_strategy='minority', random_state=0)
    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
    col_name = ['Stroke', 'Sebelum Resampling', 'Sesudah Resampling']
    row_one = [0, sum(y_train == 0), sum(y_train_resample == 0)]
    row_two = [1, sum(y_train == 1), sum(y_train_resample == 1)]
    header = 'Stroke Frequencies'
    table = SingleTable([col_name,row_one,row_two], title=header)
    print(table.table)
    return model(X_train_resample,X_test, y_train_resample, y_test)

#sinta
def model(X_train_resample,X_test, y_train_resample, y_test):
    print('Fit Data to Model...')
    print('Train the Data...')
    clf = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train_resample, y_train_resample)
    print('Data Trained...')
    Y_pred = clf.predict(X_test)
    print(f'Metrics:\n{classification_report(y_test, Y_pred)}')
    return saving_model(clf)

#fathur
def saving_model(clf):
    with open('saved_model', 'wb') as f:
        pickle.dump(clf, f)
        print('Saving Model...')
        path = os.path.abspath('saved_model')
        print(f'Model Saved: {path}')


if __name__ == '__main__':
    data()

