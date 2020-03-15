from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np

df = pd.read_csv('halp.csv')

print(df.head())

X = df.iloc[:, :-1].values
Y = df.iloc[:, 6].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = BernoulliNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn import metrics
print("GaussianNaive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


trip = int(input("Type 1 for yes and 0 for no for the following questions: Have you gone on a trip in the past month?"))
fever = int(input("Do you currently have a fever?"))
eye = int(input("Did your eyes itch in the past week?"))
nose = int(input("Do you have a runny nose?"))
breath = int(input("Have you had trouble breathing recently?"))
cough = int(input("Have you coughed much lately?"))


y_output = classifier.predict([[trip, fever, eye, nose, breath, cough]])
print(y_output)
