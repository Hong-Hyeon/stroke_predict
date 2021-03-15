import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def data_load():
    df = pd.read_csv('./data/healthcare-dataset-stroke-data.csv')
    return df

def data_prepro(df):
    data_cols = list(df.columns)

    # id 제거
    df.drop(data_cols[0], axis=1, inplace=True)

    # 성별
    # print(df['gender'].value_counts())
    df['gender'].replace({
        'Male':0,
        'Female':1,
        'Other':2
    }, inplace=True)

    # 결혼여부
    # print(df['ever_married'].value_counts())
    df['ever_married'].replace({
        'Yes':0,
        'No':1
    }, inplace=True)

    # work_type
    # print(df['work_type'].value_counts())
    # 결과
    # '''
    # Private          2925
    # Self-employed     819
    # children          687
    # Govt_job          657
    # Never_worked       22
    # '''
    work_type_cols = df['work_type'].value_counts().index.tolist()
    count = 0
    for cols in work_type_cols:
        df['work_type'].replace({
            '{}'.format(cols) : count
        }, inplace=True)
        count+=1
    # print(df['work_type'].value_counts())

    # 거주지 유형
    # print(df['Residence_type'].value_counts())
    df['Residence_type'].replace({
        'Urban':0,
        'Rural':1
    }, inplace=True)

    # bmi 지수
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    # print(df['bmi'].isnull().value_counts())

    # 흡연여부
    # print(df['smoking_status'].value_counts())
    smoking_cols = df['smoking_status'].value_counts().index.tolist()
    # print(smoking_cols)
    count = 0
    for cols in smoking_cols:
        df['smoking_status'].replace({
            '{}'.format(cols):count
        }, inplace=True)
        count+=1
    # print(df['smoking_status'].value_counts())

    # Dataframe의 label제거 및 label시리즈화
    df.drop(data_cols[-1], axis=1, inplace=True)

    # Normalization
    scaler = MinMaxScaler()
    scaled_cols = list(df.columns)
    df_scaled = scaler.fit_transform(df[scaled_cols])

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scaled_cols

    return df_scaled

def modeling():
    model = Sequential([
        Dense(124, input_shape=[10], activation='relu'),
        Dense(62, activation='relu'),
        Dropout(0.3),
        Dense(31, activation='relu'),
        Dropout(0.2),
        Dense(15, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model

if __name__ == '__main__':
    df = data_prepro(data_load())

    x_train, x_test, y_train, y_test = train_test_split(df,df.iloc[:,-1],test_size=0.4, shuffle=True)

    model = modeling()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if not os.path.exists('./data/check'):
        os.mkdir('./data/check')

    checkpath = './data/check/check.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpath,
                                 save_best_only=True,
                                 verbose=1,
                                 monitor='val_loss')

    model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[checkpoint], epochs=50, batch_size=30)

    model.save_weights(checkpath)
    print(model.evaluate(x_test, y_test))

    predict = model.predict(x_test)

    if not os.path.exists('./data/models'):
        os.mkdir('./data/models')

    model.save('./data/models/stroke_model.h5')