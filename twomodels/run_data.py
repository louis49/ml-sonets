from data import Data
from model_phon import PhonModel
from model_verset import VersetModel
from model_sonnet import SonnetModel

data = Data()
data.analyze()
data.save()
data.load()


