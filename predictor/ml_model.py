# predictor/ml_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

def train_and_save_models():
    df = pd.read_csv("C:\\Users\\danie\\Documents\\script\\Mental_Health_Lifestyle_CLS.csv")

    label_encoders = {}
    categorical_cols = ['Exercise Level', 'Stress Level', 'Diet Type']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    features = [
        'Sleep Hours',
        'Screen Time per Day (Hours)',
        'Social Interaction Score',
        'Exercise Level',
        'Stress Level',
        'Diet Type'
    ]
    target = 'Estado de Ánimo'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True)
    }

    os.makedirs('predictor/models', exist_ok=True)

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        joblib.dump(model, f'predictor/models/{name}.pkl')
    
    joblib.dump(accuracies, f'predictor/models/accuracies.pkl')

    joblib.dump(model, r'C:\Users\danie\Documents\script\Django\predictor\model.pkl')
    joblib.dump(label_encoders, r'C:\Users\danie\Documents\script\Django\predictor\encoders.pkl')

if __name__ == "__main__":
    train_and_save_models()
