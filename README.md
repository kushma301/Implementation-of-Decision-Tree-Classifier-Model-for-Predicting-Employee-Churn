
# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.KUSHMA
RegisterNumber: 212224040168 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(15,10))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)
/*
```

## Output:
<img width="1603" height="782" alt="image" src="https://github.com/user-attachments/assets/12035093-8af5-442f-b43c-c86c6cb9d612" />
<img width="1607" height="868" alt="image" src="https://github.com/user-attachments/assets/66d88812-c16e-436e-b959-c64878a67d55" />
<img width="1300" height="315" alt="image" src="https://github.com/user-attachments/assets/1793cf5b-719a-4dcb-b9d8-fd948aafda60" />
<img width="1536" height="312" alt="image" src="https://github.com/user-attachments/assets/bf8dbe1c-678c-4d1b-8bff-df7e50216e71" />
<img width="1470" height="902" alt="image" src="https://github.com/user-attachments/assets/7f7e6a42-6bcd-45a2-a1eb-eaf65ca0bb28" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
