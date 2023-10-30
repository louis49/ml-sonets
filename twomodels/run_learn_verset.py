from data import Data
from model_phon import PhonModel
from model_verset import VersetModel
from model_sonnet import SonnetModel

data = Data()
#data.analyze()
#data.save()
data.load()

verset_model = VersetModel(data)
verset_model.train(use_tuner=False)


