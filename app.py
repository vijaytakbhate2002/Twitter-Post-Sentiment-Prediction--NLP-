from config import config
import joblib
from data_manipulation.data_handling import loadData

encoder = joblib.load(config.ENCODER)

print(dir(encoder))
print(encoder.classes_)