from sklearn.pipeline import Pipeline
from config import config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data_manipulation.data_processing import NullRemoving, TextProcessing, VectorizingForPrediction
from config import config
import pandas as pd
from data_manipulation.data_handling import loadData, dumpPipeline
from config import config
from sklearn.preprocessing import LabelEncoder
import logging
import os
import joblib
logging.basicConfig(
    filename='process.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  
)

process_pipe = Pipeline([
    ("NullRemoving_initial", NullRemoving()),
    ("TextProcessing", TextProcessing(col=config.INPUT_COL[0])),
    ("Vectorizing", VectorizingForPrediction(col=config.INPUT_COL[0])),
    ("NullRemoving_end", NullRemoving()),
])

def trainPipe() -> None:
    """remove pretrained model from file with it's metadata (vectorizer and label encoder)
        """
    if os.path.exists(config.CLASSIFIER):
        os.remove(config.CLASSIFIER)
    if os.path.exists(config.VERCTORIZER):
        os.remove(config.VERCTORIZER)
    if os.path.exists(config.ENCODER):
        os.remove(config.ENCODER)

    df = loadData()
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(df[config.TARGET_COL]), name=config.TARGET_COL[0])
    joblib.dump(encoder, config.ENCODER)
    logging.info(f"columns of loaded data = {df.columns}")
    df = process_pipe.fit_transform(X=df, y=y_encoded)
    logging.info(f"after data processing columns = {df.columns}, lenght of columns = {len(df.columns)}")
    model = LogisticRegression()
    model.fit(X=df, y=y_encoded)
    dumpPipeline(model)

if __name__ == "__main__":
    trainPipe()
 