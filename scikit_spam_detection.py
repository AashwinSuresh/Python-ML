import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import accuracy_score

from sklearn.metrics import classification_report

df= pd.read_csv("datasets\spam.csv",encoding="latin1")
print(df.head())
y = df["v1"]
x = df["v2"]

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

vectorizer = TfidfVectorizer()
x_vectorized_training_set = vectorizer.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_vectorized_training_set,y_train)

x_vectorized_test_set = vectorizer.transform(x_test)

predictions = model.predict(x_vectorized_test_set)

count = 0
for pred,real in zip(predictions,y_test):
    if pred!=real:
        count+=1


print(f"TOTAL WRONG PREDICTIONS OUT OF {len(predictions)} VALUE : {count}")
accuracy = accuracy_score(y_test,predictions)
print(f"ACCURACY OF THE MODEL IS : {accuracy*100 :.2f}%\n")
print(classification_report(y_test,predictions))