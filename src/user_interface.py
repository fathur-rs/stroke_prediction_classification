import pandas as pd
import numpy as np
import pickle
import os
from terminaltables import SingleTable

path = os.path.abspath('../model/saved_model.pkl')  # Ensure this path correctly points to your pickle file
database = []

def user_input():
    try:
        data = {
            'Nama Pasien': input('Nama Pasien: ').title(),
            'Umur Pasien': int(input('Umur Pasien: ')),
            'Berat Badan Pasien (kg)': float(input('Berat Badan Pasien (kg): ')),
            'Tinggi Badan Pasien (cm)': float(input('Tinggi Badan Pasien (cm): ')),
            'Rata-rata Kadar Gula Darah Pasien': float(input('Rata-rata Kadar Gula Darah Pasien: ')),
            'Gender Pasien': int(input('Gender Pasien [FEMALE (0) / MALE (1)]: ')),
            'Tipe Profesi Pasien': int(input('Tipe Profesi Pasien [GOV JOB (0) / NEVER WORKED (1) / PRIVATE (2) / SELF-EMP (3) / CHILDREN (4)]: ')),
            'Kawasan Tempat Tinggal Pasien': int(input('Kawasan Tempat Tinggal Pasien [RURAL (0) / URBAN (1)]: ')),
            'Apakah Pasien Sudah Menikah?': int(input('Apakah Pasien Sudah Menikah? [NO (0) / YES (1)]: ')),
            'Apakah Pasien Perokok?': int(input('Apakah Pasien Perokok? [UNKNOWN (0), FORMERLY (1), NO (2), YES (3)]: ')),
            'Apakah Pasien Memiliki Penyakit Hipertensi?': int(input('Apakah Pasien Memiliki Penyakit Hipertensi? [NO (0) / YES (1)]: ')),
            'Apakah Pasien Memiliki Penyakit Jantung?': int(input('Apakah Pasien Memiliki Penyakit Jantung? [NO (0) / YES (1)]: '))
        }
        bmi = round(data['Berat Badan Pasien (kg)'] / (data['Tinggi Badan Pasien (cm)'] / 100) ** 2, 3)
        return create_dataframe(data, bmi)
    except Exception as e:
        print(f"Error: {e}")
        return user_input()

def create_dataframe(data, bmi):
    patient_data = {
        'Name': data['Nama Pasien'],
        'gender': data['Gender Pasien'],
        'age': data['Umur Pasien'],
        'hypertension': data['Apakah Pasien Memiliki Penyakit Hipertensi?'],
        'heart_disease': data['Apakah Pasien Memiliki Penyakit Jantung?'],
        'ever_married': data['Apakah Pasien Sudah Menikah?'],
        'work_type': data['Tipe Profesi Pasien'],
        'Residence_type': data['Kawasan Tempat Tinggal Pasien'],
        'avg_glucose_level': data['Rata-rata Kadar Gula Darah Pasien'],
        'bmi': bmi,
        'smoking_status': data['Apakah Pasien Perokok?']
    }
    df = pd.DataFrame([patient_data])
    return predict(df)

def predict(df):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        prediction = model.predict(df.drop('Name', axis=1))[0]
        result = 'Kemungkinan Besar Tidak Stroke' if prediction == 0 else 'Kemungkinan Besar Stroke'
        return finalize_data(df, result)

def finalize_data(df, prediction):
    df['Result'] = prediction
    df.replace({
        'gender': {0: 'Female', 1: 'Male'},
        'ever_married': {0: 'Unmarried', 1: 'Married'},
        'Residence_type': {0: 'Rural', 1: 'Urban'},
        'hypertension': {0: 'Negative', 1: 'Positive'},
        'heart_disease': {0: 'Negative', 1: 'Positive'},
        'work_type': {0: 'Govt. Job', 1: 'Never Worked', 2: 'Private', 3: 'Self Employed', 4: 'Children'},
        'smoking_status': {0: 'Unknown', 1: 'Formerly', 2: 'Negative', 3: 'Positive'}
    }, inplace=True)
    database.append(df.iloc[0].to_dict())
    return display_data()

def display_data():
    if not database:
        print("No data available.")
        return

    table_data = [list(database[0].keys())]
    table_data += [list(record.values()) for record in database]
    table = SingleTable(table_data, title='Stroke Prediction')
    print(f'\n{table.table}\n')
    return option()

def option():
    choice = input('Input More Data? (Yes or No): ').strip().title()
    if choice == 'Yes':
        return user_input()
    elif choice == 'No':
        print('Terimakasih Telah Menggunakan Program ^_^')
    else:
        print('Invalid Input!')
        return option()

if __name__ == '__main__':
    user_input()
