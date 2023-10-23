from data import Data
from phon_model import PhonModel

data = Data()
#data.analyze()
#data.save()
data.load()

phon_model = PhonModel(data)
phon_model.train(use_tuner=True)



