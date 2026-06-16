from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from termcolor import colored
from tabulate import tabulate
import os
import time 
import joblib


iris_dataset = load_iris()
x = iris_dataset.data
y = iris_dataset.target

print(len(x),type(x),x[0])
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    # random_state = 42
)

print(len(x_train),x_train[0])
print(len(x_test),x_test[0])
print(len(y_train),y_train[0])
print(len(y_test),y_test[0])

model = LogisticRegression()

print("\nMODEL TRAINING STARTED")
print("----------------------")

s_time = time.perf_counter()
model.fit(x_train,y_train)
e_time = time.perf_counter()
t_time=e_time-s_time

print(f"\MODEL TRAINING FINISHED \n TIME TAKEN : {t_time:.5f} ")


print("\nMODEL TESTING")
print("-------------\n")

predictions = model.predict(x_test)
result = []
for x_val , y_val , pred in zip(x_test,y_test,predictions):
    result.append(colored("True","blue") if y_val == pred else colored("False","red",attrs=["bold","underline"]))

data =zip(x_test,y_test,predictions,result)
headers = ["X_VALUE","Y_VALUE","PREDICTIONS","RESULT"]

table = tabulate(data,headers,"grid")

print(table)
accuracy = accuracy_score(y_test,predictions)
print(f"THE ACCURACY OF THE MODEL IS : {accuracy}")

model_path=r".\models\iris_model.pkl"
print(f"saving model into location : {model_path}")
if not os.path.exists(model_path):
    os.makedirs("models")
joblib.dump(model,model_path)
print("succesfull")

