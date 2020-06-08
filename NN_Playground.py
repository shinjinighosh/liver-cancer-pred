#!/usr/bin/python3

import warnings
import pandas as pd
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from tensorflow.keras.preprocessing.text import Tokenizer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data = pd.read_csv('datasets_2607_4342_indian_liver_patient_labelled.csv')

# creating input features and labels
X = data[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
          'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
Y = data[['Dataset']]  # labels

# preprocessing


# building model
model = Sequential()
input_dims = X.shape  # should be (583, 10)
model.add(Dense(10, activation='relu', input_dim=input_dims[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting model on data
model.fit(X, Y, epochs=10, batch_size=20)
