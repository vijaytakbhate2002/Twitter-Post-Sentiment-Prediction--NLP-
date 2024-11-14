import sys
ROOT_PATH = "\\".join(__file__.split('\\')[:-2])
sys.path.append(ROOT_PATH)
import os

# Data configuration (processed data)
DATA_PATH = os.path.join(ROOT_PATH, "notebooks\\data_pipeline\\data\\vectorized_data")
X_DATA = os.path.join(DATA_PATH, "Vectorized_X.csv")
Y_DATA = os.path.join(DATA_PATH, "Vectorized_y.csv")

# row data configuration 
ROW_X_DATA = os.path.join("row_data\\twitter_training.csv")
ROW_Y_DATA = os.path.join("row_data\\twitter_validation.csv")

# Model paths
CLASSIFIER = "trained_models\\classifier.pkl"
VERCTORIZER = "trained_models\\vectorizer.pkl"
ENCODER = "trained_models\\encoder.pkl"

# input columns
INPUT_COL = ['tweets']
TARGET_COL = ['@sentiments']
PROCESS_COLUMNS = ['tweets', '@sentiments']

# Shrinked data

TRAINING = "notebooks\\data_pipeline\\data\\shrinked_data\\testing.csv"
TESTING = "notebooks\\data_pipeline\\data\\shrinked_data\\training.csv"

# ETL OUTPUT DATA
ETL_OUT_DATA = "notebooks/data_pipeline/data/processed_data"

ETL_OUT_TEST_DATA = os.path.join(ETL_OUT_DATA, "testing.csv")
ETL_OUT_TRAIN_DATA = os.path.join(ETL_OUT_DATA,"training.csv")


# columns configuration (ETL)
ETL_COLS = ["topics", "sentiments", "tweets"]
ETL_TARGET = 'sentiments'
ETL_DATA_FORMAT = 'csv'

# metrics used for model building
EVALUATION_MATRICES = [
        "train_score",
        "test_score",
        "precision",
        "recall",
        "f1"
        ]

MODEL_NAEM = "LogisticRegression"
BEST_PARAMS = params = {
    "C": 0.1,
    "Penalty": "l2",
    "Solver": "newton-cg",
    "Max_iter": 100
}

# vectorization technique
VECTORIZATION = 'tf-idf'