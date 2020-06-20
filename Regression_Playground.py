#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

data = pd.read_csv("datasets_2607_4342_indian_liver_patient_labelled.csv")
df = pd.DataFrame(data)

for col in df.columns:
    df[col] = df[col].fillna(0)

# Linear Regression
headers = list(df.columns)
headers.remove('Dataset')
headers.remove('Gender')
X = df[headers]
Y = df['Dataset']

linregmodel = linear_model.LinearRegression()
linregmodel.fit(X, Y)

# print("Intercept is %f" % linregmodel.intercept_)
# print("Coefficients are", linregmodel.coef_)
# print("R2 score is", linregmodel.score(X, Y))

Y_pred = linregmodel.predict(X)

# Logistic Regression
df['intercept'] = linregmodel.intercept_
df['Dataset'] = df['Dataset'].replace([1], 0)
df['Dataset'] = df['Dataset'].replace([2], 1)

logregmodel = sm.Logit(Y, X)
result = logregmodel.fit()
print(result.summary())
print("Confidence intervals are", result.conf_int())

Y_pred = logregmodel.predict(X.T)

cnf_matrix = confusion_matrix(Y, Y_pred[:, 0:1])
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# everything seems to be labelled 1?
print("Accuracy is thus", (416-167) / 416)

