from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train, dataset_name):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/churn_model_{dataset_name}.pkl')
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {"accuracy": accuracy}
