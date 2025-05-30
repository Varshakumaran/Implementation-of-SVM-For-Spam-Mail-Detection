# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, scikit-learn modules for model building, and evaluation.

2. Load the dataset using pandas read_csv() with encoding set to 'Windows-1252'.

3. Display the first few rows and check the shape of the dataset.

4. Extract input features (text messages) and output labels (spam/ham) from the dataset.

5. Split the data into training and testing sets using train_test_split().

6. Initialize CountVectorizer to convert text data into numerical form.

Fit the vectorizer on the training data and transform both training and test data.

Initialize the Support Vector Machine (SVM) classifier using the SVC class.

Train the SVM model using the training data with the fit() method.

Make predictions on the test data using the predict() method.

Evaluate the model using:

a. accuracy_score() for accuracy,

b. confusion_matrix() for confusion matrix,

c. classification_report() for precision, recall, and F1-score.

Print the predicted output, accuracy, confusion matrix, and classification report.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Varsha k
RegisterNumber:  212223220122
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("/content/spam.csv", encoding='Windows-1252')
print(df)

print("Dataset Shape:", df.shape)

messages = df['v2'].values
labels = df['v1'].values
print("Messages shape:", messages.shape)
print("Labels shape:", labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(messages, labels, test_size=0.2, random_state=0)
X_train
X_train.shape

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
svm_model = SVC()
svm_model.fit(X_train_vector, Y_train)
Y_pred = svm_model.predict(X_test_vector)
print("Predicted Output:\n", Y_pred)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(Y_test, Y_pred)
print("Classification Report:\n", class_report)
```
## Output:
## Displaying the Dataset
![image](https://github.com/user-attachments/assets/899d6319-6ff9-42d1-b7bb-30062b8446aa)

## df.shape
![image](https://github.com/user-attachments/assets/f9bf0182-0281-4b17-961c-1b6f6defbdd7)
## Printing the shape of Messages and Labels
![image](https://github.com/user-attachments/assets/06ee9ecc-1e6d-4257-817f-6ef513006958)
## Displaying X_train and it's shape
![image](https://github.com/user-attachments/assets/9062635e-7618-4ace-af2c-7839f55887e5)
## svm_model
![image](https://github.com/user-attachments/assets/b040d176-5b73-4d57-80bc-cd1ced929e71)
## Predicting the output
![image](https://github.com/user-attachments/assets/e58a5798-5a6c-467f-aa0f-b5f6229e63a4)
## Evaluating the Accuracy
![image](https://github.com/user-attachments/assets/880a7ee8-6e8b-44ed-950f-880a16c6dc62)
## Confusion Matrix
![image](https://github.com/user-attachments/assets/71ae0dfc-97b2-4fa7-805f-ff294b5f9b14)
## Classification Report
![image](https://github.com/user-attachments/assets/500197b0-3e80-42ee-9726-4514a5cf8893)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
