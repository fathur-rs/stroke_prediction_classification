import pandas as pd
import numpy as np
import pickle
import os
from terminaltables import SingleTable

"""Jalankan module training.py terlebih dahulu"""

# lokasi file pickle yang sebelumnya sudah kita buat
path = os.path.abspath('module_decision')
# empty list untuk menyimpan data user input
database = []

def user_input():
    # user input
    try:
        input_name = input('Nama: ').title()
        input_age = int(input('Umur: '))
        input_weight = float(input('Berat Badan Anda (kg): '))
        input_height = float(input('Tinggi Badan Anda (cm): '))
        input_glukosa = float(input('Rata-rata Kadar Gula Darah: '))
        input_gender = int(input('Gender [FEMALE (0) / MALE (1)]: '))
        input_work_type = int(input('Tipe Profesi [GOVERMENT JOB (0) / NEVER WORKED (1) / PRIVATE (2) / SELF-EMP (3) / STUDENTS (4)]: '))
        input_residence_type = int(input('Kawasan Tempat Tinggal [RURAL (0) / URBAN (1)]: '))
        input_ever_married = int(input('Apakah Anda Sudah Menikah? [NO (0) / YES (1)]: '))
        input_smoking = int(input('Apakah Anda Perokok? [UNKNOWN (0), FORMERLY (1), NO (2), YES (3)]: '))
        input_hypertension = int(input('Apakah Anda Memiliki Penyakit Hipertensi? [NO (0) / YES (1)]: '))
        input_heart_disease = int(input('Apakah Anda Memiliki Penyakit Jantung? [NO (0) / YES (1)]: '))
        bmi = round(input_weight/pow((input_height/100), 2),3)
        return dataframe_maker(input_name, input_gender,input_age, input_hypertension, input_heart_disease, input_ever_married, input_work_type,input_residence_type,input_glukosa, bmi,
                               input_smoking)
    except Exception as error:
        print(error)
        return user_input()

def dataframe_maker(*user_input):
    # inisiasi nama column
    columns = ['Name', 'Gender', 'Age','Hypertension', 'Heart Disease', 'Married Status', 'Work Type', 'Residence Type', 'Glucose Level', 'Body Mass Index', 'Smoking Status']

    # membuat dictionary
    data_user = {col:[val] for col, val in zip(columns, user_input)}

    # dictionary to dataframe
    df = pd.DataFrame(data_user)
    return predict(df, data_user)

def predict(df, data_user):
    # kita import file pickle nya
    with open(path,'rb') as f:
        mp = pickle.load(f)
        # langsung kita prediksi output dari user input
        Y_pred2 = ''.join(np.where(mp.predict(df.iloc[:, 1:]) == 0, 'Kemungkinan Besar Tidak Stroke', 'Kemungkinan Besar Stroke'))
        return binary_to_string(df, data_user, Y_pred2)

def binary_to_string(df, data_user, Y_pred2):
    # mengubah value dictionary yang berupa list menjadi str
    data_user_interface = {k: str(v[0]) for k, v in data_user.items()}

    # mengubah kembali tipe data binary/integer menjadi string/category dalam dictionary yang sudah di ubah
    data_user_interface['Result'] = Y_pred2
    data_user_interface['Hypertension'] = ''.join(np.where(df.Hypertension.values == 0, 'Negative', 'Positive'))
    data_user_interface['Gender'] = ''.join(np.where(df.Gender == 0, 'Female', 'Male'))
    data_user_interface['Married Status'] = ''.join(np.where(df['Married Status'] == 0, 'Unmarried', 'Married'))
    data_user_interface['Residence Type'] = ''.join(np.where(df['Residence Type'] == 0, 'Rural', 'Urban'))
    data_user_interface['Heart Disease'] = ''.join(np.where(df['Heart Disease'] == 0, 'Negative', 'Positive'))
    data_user_interface['Work Type'] = ''.join(df['Work Type'].map({0: 'Govt. Job', 1: 'Never Worked', 2: 'Private', 3: 'Self Employed', 4: 'Student'}))
    data_user_interface['Smoking Status'] = ''.join(df['Smoking Status'].map({0: 'Unknown', 1: 'Formerly', 2: 'Negative', 3: 'Positive'}))
    return database.append(data_user_interface), user_interface()

def user_interface():
    # unpacking dictionary untuk membuat tabel
    cols, rows = [], []
    for data in database:
        col, row = [], []
        for m, n in data.items():
            col.append(m)
            row.append(n)
        cols.append(col)
        rows.append(row)
    # membuat tabel
    data_tabel = [cols[0], *rows]
    title = 'Stroke Prediction'
    table = SingleTable(data_tabel, title)
    print(f'\n{table.table}\n')
    return option()

def option():
    # input more data?
    prompt = input('Input More Data? (Yes or No): ').title()
    if prompt == 'Yes':
        user_input()
    elif prompt =='No':
        print('Terimakasih Telah Menggunakan Program ^_^')
    else:
        print('Invalid Input!')
        return option()

if __name__ == '__main__':
    # jalankan function user_input()
    user_input()
