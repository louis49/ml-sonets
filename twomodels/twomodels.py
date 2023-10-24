from data import Data
from phon_model import PhonModel
from verset_model import VersetModel

data = Data()
data.analyze()
data.save()
data.load()

#phon_model = PhonModel(data)
#phon_model.train(use_tuner=True)

verset_model = VersetModel(data)
verset_model.train(use_tuner=False)



