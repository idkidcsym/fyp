from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib
from config import MODEL_PATH

MODEL_PATH = 'saved_models/'

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH + 'random_forest.pkl')
    return model

def load_model():
    return joblib.load(MODEL_PATH + 'random_forest.pkl')
