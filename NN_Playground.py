#!/usr/bin/python3

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from math import sqrt


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
Y_numeric = data[['Dataset']]
Y = to_categorical(data[['Dataset']])
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# encoded_Y = to_categorical(encoded_Y)

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

# print model history keys for debugging
# print(history.history.keys())

# create accuracy plots
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy.png')
# plt.show()

# create loss plots
plt.figure()
ax2 = plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss.png')
# plt.show()


# obtain confusion matrix
prediction = model.predict_classes(X)
conf_matrix = confusion_matrix(Y_numeric, prediction)  # np.argmax(prediction, axis=0)
print('Confusion Matrix')
print(conf_matrix)
(tn, fp), (fn, tp) = conf_matrix

# plot confusion matrix
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="viridis", fmt='g')
ax.xaxis.set_label_position("bottom")
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('Confusion_matrix.png')
plt.show()

# calculating more metrics
error_rate = (fp + fn) / (tp + tn + fp + fn)
accuracy = 1 - error_rate
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
false_positive_rate = fp / (tn + fp)
mcc = ((tp * tn) - (fp * fn)) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
beta = 1
f_score = ((1 + beta * beta) * (precision * sensitivity)) / (beta * beta * precision + sensitivity)
metrics = {'Error rate': error_rate, "Accuracy": accuracy, "Loss": loss,
           "Sensitivity": sensitivity, "Specificity": specificity, "Precision": precision, "FPR": false_positive_rate, "MCC": mcc, "F score": f_score}

print('=========================')
print("        Metrics")
print('=========================')
# print(pd.DataFrame(metrics, index=[0]).T)
print(pd.Series(metrics).T)
