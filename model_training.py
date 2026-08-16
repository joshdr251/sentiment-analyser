
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load file
while True:
    try:
        filepath = "imdb_dataset_50000_cleaned.csv"
        df = pd.read_csv(filepath)
        break
    except FileNotFoundError:
            print("File not found, try again.")
            continue


# separate features x and target variable y
X = df["review"]
y = df["sentiment"]

# dividing in training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify ensures that training data is 50/50 (positive/negative)

# creating vectorizer
vectorizer = TfidfVectorizer(max_features=50000)
x_train_vec = vectorizer.fit_transform(X_train)  # learns from the data
x_test_vec = vectorizer.transform(X_test)  # only using, not learning

# training the model
model = LogisticRegression(max_iter=5000)
model.fit(x_train_vec, y_train)

# predicting and evaluation
y_pred = model.predict(x_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print()
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model, "logistic_regression_model.pkl")
