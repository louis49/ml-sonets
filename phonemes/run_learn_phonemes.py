from phonemes.learn_phonemes.data_phonemes import Data
from phonemes.learn_phonemes.learn_phonemes import Learn

data = Data()
data.load()

learn = Learn(data)
learn.train(tuner=False)
