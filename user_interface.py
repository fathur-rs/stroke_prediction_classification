import pandas as pd
import numpy as np
import pickle
import os
from terminaltables import SingleTable

path = os.path.abspath('saved_model')
database = []

def user_input():
    try:
        input_name = input('Nama Pasien: ').title()
        input_age = int(input('Umur Pasien: '))
        input_weight = float(input('Berat Badan Pasien (kg): '))
        input_height = float(input('Tinggi Badan Pasien (cm): '))
        input_glukosa = float(input('Rata-rata Kadar Gula Darah Pasien: '))
        input_gender = int(input('Gender Pasien [FEMALE (0) / MALE (1)]: '))
        input_work_type = int(input('Tipe Profesi Pasien [GOVERMENT JOB (0) / NEVER WORKED (1) / PRIVATE (2) / SELF-EMP (3) / CHILDREN (4)]: '))
        input_residence_type = int(input('Kawasan Tempat Tinggal Pasien [RURAL (0) / URBAN (1)]: '))
        input_ever_married = int(input('Apakah Pasien Sudah Menikah? [NO (0) / YES (1)]: '))
        input_smoking = int(input('Apakah Pasien Perokok? [UNKNOWN (0), FORMERLY (1), NO (2), YES (3)]: '))
        input_hypertension = int(input('Apakah Pasien Memiliki Penyakit Hipertensi? [NO (0) / YES (1)]: '))
        input_heart_disease = int(input('Apakah Pasien Memiliki Penyakit Jantung? [NO (0) / YES (1)]: '))
        bmi = round(input_weight/pow((input_height/100), 2),3)
        return dataframe_maker(input_name, input_gender,input_age, input_hypertension, input_heart_disease, input_ever_married,
                               input_work_type,input_residence_type,input_glukosa, bmi,input_smoking)

    except Exception as error:
        print(error)
        return user_input()

def dataframe_maker(*user_input):
    columns = ['Name', 'gender', 'age','hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    data_user = {col:[val] for col, val in zip(columns, user_input)}
    df = pd.DataFrame(data_user)
    return predict(df, data_user)

def predict(df, data_user):
    with open(path,'rb') as f:
        mp = pickle.load(f)
        Y_pred2 = ''.join(np.where(mp.predict(df.iloc[:, 1:]) == 0, 'Kemungkinan Besar Tidak Stroke', 'Kemungkinan Besar Stroke'))
        return binary_to_string(df, data_user, Y_pred2)

def binary_to_string(df, data_user, Y_pred2):
    data_user_interface = {k: str(v[0]) for k, v in data_user.items()}
    data_user_interface['Result'] = Y_pred2
    data_user_interface['hypertension'] = ''.join(np.where(df['hypertension'].values == 0, 'Negative', 'Positive'))
    data_user_interface['gender'] = ''.join(np.where(df['gender'] == 0, 'Female', 'Male'))
    data_user_interface['ever_married'] = ''.join(np.where(df['ever_married'] == 0, 'Unmarried', 'Married'))
    data_user_interface['Residence_type'] = ''.join(np.where(df['Residence_type'] == 0, 'Rural', 'Urban'))
    data_user_interface['heart_disease'] = ''.join(np.where(df['heart_disease'] == 0, 'Negative', 'Positive'))
    data_user_interface['work_type'] = ''.join(df['work_type'].map({0: 'Govt. Job', 1: 'Never Worked', 2: 'Private', 3: 'Self Employed', 4: 'Children'}))
    data_user_interface['smoking_status'] = ''.join(df['smoking_status'].map({0: 'Unknown', 1: 'Formerly', 2: 'Negative', 3: 'Positive'}))
    return database.append(data_user_interface), user_interface()

def user_interface():
    cols, rows = [], []
    for data in database:
        col, row = [], []
        for m, n in data.items():
            col.append(m)
            row.append(n)
        cols.append(col)
        rows.append(row)
    data_tabel = [cols[0], *rows]
    title = 'Stroke Prediction'
    table = SingleTable(data_tabel, title)
    print(f'\n{table.table}\n')
    return option()

def option():
    prompt = input('Input More Data? (Yes or No): ').title()
    if prompt == 'Yes':
        user_input()
    elif prompt =='No':
        print('Terimakasih Telah Menggunakan Program ^_^')
    else:
        print('Invalid Input!')
        return option()

if __name__ == '__main__':
    user_input()
