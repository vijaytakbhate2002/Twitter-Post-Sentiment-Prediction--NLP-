from train_pipe import process_pipe
from data_manipulation.data_handling import loadData
from config import  config
import pandas as pd
import joblib

def newDataPredictor(X:pd.DataFrame) -> list[str]:
    """Predict for given input and return list with predicted classes"""
    id2label = {}
    try:
        classifier = joblib.load(config.CLASSIFIER)
        labels = joblib.load(config.ENCODER).classes_
        for i, label in enumerate(labels):
            id2label[i] = label

    except: 
        raise FileNotFoundError("trained classifier not found train pipe before prediction")
    X = process_pipe.fit_transform(X=X, y=None)
    predictions = classifier.predict(X)
    predictions = [id2label[val] for val in predictions]
    return predictions

if __name__ == "__main__":
    df = loadData()
    df = df[config.INPUT_COL][:50]
    print(newDataPredictor(X=df))