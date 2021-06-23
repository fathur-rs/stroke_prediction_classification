import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import time
from sklearn.metrics import classification_report
from terminaltables import SingleTable
import os

def data():
    url = 'https://raw.githubusercontent.com/fathur-rs/uas/master/healthcare-dataset-stroke-data.csv'
    data = pd.read_csv(url)
    print('Reading Dataset...'), time.sleep(1)
    return data_clean(data)

def data_clean(data):
    df = data.loc[data["gender"] != 'Other']
    df.dropna(axis=0, inplace=True)
    return data_prep(df)

def data_prep(df):
    label_encoder = preprocessing.LabelEncoder()
    for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        df[i] = label_encoder.fit_transform(df[i])
    print('Data Preprocessing...'), time.sleep(1)
    return splitting_data(df)

def splitting_data(df):
    X = df.drop(['id', 'stroke'], axis=1)
    Y = df.stroke
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.29, random_state=42)
    print('Split the data...'), time.sleep(1)
    return resampling(X_train, X_test, y_train, y_test)

def resampling(X_train, X_test, y_train, y_test):
    print('Resampling Train Data...'), time.sleep(1)
    smote = SMOTE(sampling_strategy='minority', random_state=0)

    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
    table = SingleTable([
        ['Stroke', 'Sebelum Resampling', 'Sesudah Resampling'],
        [0, sum(y_train == 0), sum(y_train_resample == 0)],
        [1, sum(y_train == 1), sum(y_train_resample == 1)]
    ], title='Y_Train')
    print(table.table), time.sleep(2)

    return model(X_train_resample,X_test, y_train_resample, y_test)

def model(X_train_resample,X_test, y_train_resample, y_test):
    print('Fit Data to Model...'), time.sleep(1)
    print('Train the Data...'), time.sleep(1)
    clf = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train_resample, y_train_resample)
    print('Data Trained...'), time.sleep(1)

    Y_pred = clf.predict(X_test)
    print(f'Metrics:\n{classification_report(y_test, Y_pred)}'), time.sleep(2)
    return saving_model(clf)

def saving_model(clf):
    with open('module_decision', 'wb') as f:
        pickle.dump(clf, f)
        print('Saving Model...'), time.sleep(1)
        path = os.path.abspath('module_decision')
        print(f'Model Saved: {path}'), time.sleep(1)


if __name__ == '__main__':
    data()

