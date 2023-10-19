from tensorflow.python.keras.callbacks import Callback
class EpochCallback(Callback):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

