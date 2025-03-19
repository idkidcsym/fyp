from data_loader import load_and_preprocess_data
from model import train_model, load_model
from evaluation import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    model = load_model()  # Load the model to ensure it's saved and loaded correctly
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(cm)

if __name__ == '__main__':
    main()
