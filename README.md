# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries NumPy, Matplotlib and Scikit-learn.
2. Create the dataset with student details and placement status.
3. Train the Logistic Regression model using the training data.
4. Predict the placement status of a new student.
5. Display the predicted result as Placed or Not Placed.
6. Evaluate the model accuracy.

## Program:
```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data (1).csv")
print("First 5 rows of the dataset:")
print(data.head())

print("\nNull values in dataset:\n", data.isnull().sum())

data = data.drop(['sl_no', 'salary'], axis=1)

le = LabelEncoder()
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
![alt text](image.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
