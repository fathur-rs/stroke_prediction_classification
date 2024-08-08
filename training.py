import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from terminaltables import SingleTable
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable chained assignment warning
pd.options.mode.chained_assignment = None

def load_data(url):
    """ Load dataset from a URL """
    df = pd.read_csv(url)
    logging.info('📥 Dataset loaded.')
    return df

def preprocess_data(df):
    """ Preprocess the data: filter, encode, and split """
    df = df[df["gender"] != 'Other']
    df.dropna(inplace=True)

    # Encoding categorical columns
    label_encoder = preprocessing.LabelEncoder()
    obj_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df[obj_cols] = df[obj_cols].apply(lambda x: label_encoder.fit_transform(x))

    X = df.drop(['id', 'stroke'], axis=1)
    y = df['stroke']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=42)
    logging.info('🔄 Data preprocessed and split.')
    return X_train, X_test, y_train, y_test

def resample_data(X_train, y_train):
    """ Resample the training data using SMOTE """
    logging.info('♻️ Resampling train data...')
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Displaying resampling results in a table
    summary = [
        ['Stroke', 'Before Resampling', 'After Resampling'],
        [0, sum(y_train == 0), sum(y_train_resampled == 0)],
        [1, sum(y_train == 1), sum(y_train_resampled == 1)]
    ]
    table = SingleTable(summary, title='Stroke Frequencies')
    logging.info('\n' + table.table)

    return X_train_resampled, y_train_resampled

def train_model(X_train, y_train, X_test, y_test):
    """ Train the decision tree classifier and print classification report """
    logging.info('🚀 Training model...')
    clf = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    logging.info(f'📊 Metrics:\n{classification_report(y_test, y_pred)}')
    return clf

def save_model(clf, filename='saved_model.pkl'):
    """ Save the trained model as a pickle file """
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    logging.info(f'💾 Model saved at: {os.path.abspath(filename)}')

def main():
    url = 'https://raw.githubusercontent.com/fathur-rs/uas/master/healthcare-dataset-stroke-data.csv'
    df = load_data(url)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_resampled, y_train_resampled = resample_data(X_train, y_train)
    clf = train_model(X_train_resampled, y_train_resampled, X_test, y_test)
    save_model(clf)

if __name__ == '__main__':
    main()
