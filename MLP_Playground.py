#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# loading the data
data = pd.read_csv("datasets_2607_4342_indian_liver_patient_labelled.csv")
df = pd.DataFrame(data)


# preprocessing
for col in df.columns:
    df[col] = df[col].fillna(0)
headers = list(df.columns)
headers.remove('Dataset')
df = pd.concat([df, pd.get_dummies(data['Gender'], prefix='Gender')], axis=1)
headers.remove('Gender')
df['Dataset'] = df['Dataset'].replace([1], 0)
df['Dataset'] = df['Dataset'].replace([2], 1)

# creating training data
X = df[headers]
Y = df['Dataset']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# creating the model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,),
                    random_state=1, max_iter=2000)
clf.fit(X, Y)


# predicting using MLP Model
Y_pred = clf.predict(X)
print("The loss is %.2f" % (clf.loss_))
print("Mean accuracy is %.2f" % (clf.score(X, Y)))

# comparing models
comparison_dict = {}
solvers = ['lbfgs', 'sgd', 'adam']
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
hidden_layer_sizes = [(2,), (3,), (5,), (2, 2,), (3, 2,), (5, 2,), (5, 3,), (5, 5,)]
# learning_rates = ['constant', 'invscaling', 'adaptive'] # only for sgd


print("Comparing models...")
count = 1
for solver in solvers:
    for ac in activation_functions:
        for hidden_layer_config in hidden_layer_sizes:
            print("On model #%d" % (count))
            clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_config,
                                random_state=1, max_iter=15000, activation=ac)
            clf.fit(X, Y)
            comparison_dict[count] = {"solver": solver, "activation_function": ac,
                                      "hidden_layers": hidden_layer_config, "accuracy": clf.score(X, Y), "loss": clf.loss_}
            count += 1

comparison_df = pd.DataFrame(comparison_dict).T
comparison_df['accuracy'] = pd.to_numeric(comparison_df['accuracy'])
max_acc_index = comparison_df['accuracy'].idxmax()
row = comparison_df.loc[max_acc_index]
print("The best accuracy out of %d models is for" % (count - 1))
print(row)
