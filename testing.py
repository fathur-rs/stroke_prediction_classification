import pandas as pd
pd.options.mode.chained_assignment = None
import training
import time

def data():
    url = 'https://raw.githubusercontent.com/fathur-rs/uas/master/healthcare-dataset-stroke-data%20(2).csv'
    data = pd.read_csv(url)
    print('Reading Dataset...'), time.sleep(1)
    return data_clean(data)

def data_clean(data):
    df = data.loc[data["gender"] != 'Other']
    df.dropna(axis=0, inplace=True)
    obj_df = df.select_dtypes(include=['object']).copy()
    return data_prep(obj_df, df)

def data_prep(obj_df, df):
    df = df
    for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        obj_df[col] = obj_df[col].astype('category')
        obj_df[col] = obj_df[col].cat.codes
    print('Data Preprocessing...'), time.sleep(1)
    return data_to_model(obj_df, df)

def data_to_model(obj_df, df):
    df_final = df.iloc[:, [0, 2, 3, 4, 8, 9, 11]].join(obj_df)
    return training.splitting_data(df_final)

if __name__ == '__main__':
    data()

