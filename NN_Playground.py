#!/usr/bin/python3

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    from tensorflow.keras.preprocessing.text import Tokenizer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data = pd.read_csv('datasets_2607_4342_indian_liver_patient_labelled.csv')

# preprocessing
headers = list(data.columns)
headers.remove('Dataset')
for col in data.columns:
    data[col] = data[col].fillna(0)
data = pd.concat([data, pd.get_dummies(data['Gender'], prefix='Gender')], axis=1)
headers.remove('Gender')

data['Dataset'] = data['Dataset'].replace([1], 0)
data['Dataset'] = data['Dataset'].replace([2], 1)

# creating input features and labels
X = data[headers]
Y = to_categorical(data[['Dataset']])

# building model
model = Sequential()
input_dims = X.shape  # should be (583, 10)
model.add(Dense(10, activation='relu', input_dim=input_dims[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting model on data
history = model.fit(X, Y, epochs=100, batch_size=10, validation_split=0.2)

# evaluating the model
loss, accuracy = model.evaluate(X, Y)
print('Loss on training data: %.2f' % (loss))
print('Accuracy on training data: %.2f' % (accuracy * 100))

# inspecting model
print(model.summary())

# print model history keys
print(history.history.keys())

# create accuracy plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy.png')
plt.show()

# create loss plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss.png')
plt.show()


# obtain confusion matrix
prediction = model.predict_classes(X)
print(type(prediction), prediction.shape)
print(prediction)
cm1 = confusion_matrix(Y, np.argmax(prediction, axis=0))
print('Confusion Matrix : \n')
print(cm1)


score, acc = model.evaluate(X, Y)
print('Test score:', score)
print('Test accuracy:', acc)
