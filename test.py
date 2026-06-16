import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from termcolor import colored
from sklearn.metrics import accuracy_score
dataset = load_iris()
x = dataset.data
y = dataset.target


path="C:\Source_Code\Python ML\models\iris_model.pkl"
model=joblib.load(path)

predictions = model.predict(x)

result = []
for x_val , y_val , pred in zip(x,y,predictions):
    result.append(colored("True","blue") if y_val == pred else colored("False","red",attrs=["bold","underline"]))

data =zip(x,y,predictions,result)
headers = ["X_VALUE","Y_VALUE","PREDICTIONS","RESULT"]

table = tabulate(data,headers,"grid")

print(table)
accuracy = accuracy_score(y,predictions)
print(f"THE ACCURACY OF THE MODEL IS : {accuracy}")