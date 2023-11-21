from data import Data
from model_verset import VersetModel

data = Data()
data.load()

verset_model = VersetModel(data)
verset_model.train(use_tuner=True)


