import pandas as pd
pd.options.mode.chained_assignment = None
import training
import time
from sklearn import preprocessing

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
    return training.splitting_data(df)

if __name__ == '__main__':
    data()

