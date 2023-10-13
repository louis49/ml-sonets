from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import Callback
import numpy as np

class GenerateSample(Callback):

    def __init__(self, input_text, tokenizer, total_words, max_decoder_seq_length=100, max_encoder_seq_length=100, every_epoch=1):
        super(GenerateSample, self).__init__()
        self.input_text = input_text
        self.tokenizer = tokenizer
        self.total_words = total_words
        self.max_decoder_seq_length = max_decoder_seq_length
        self.max_encoder_seq_length = max_encoder_seq_length
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0:
            generated_text = self.generate_text(self.model)
            print("\nSample generation at epoch {}: \n{}".format(epoch + 1, generated_text))

    def generate_text(self, model):
        # Tokenization
        input_seq = self.tokenizer.texts_to_sequences([self.input_text])
        encoder_input_data = pad_sequences(input_seq, maxlen=self.max_encoder_seq_length, padding='post')

        # Commencer la séquence de sortie avec le token de début
        curr_seq = [self.tokenizer.word_index['<sonnet>']]
        generated_text = []

        for i in range(self.max_decoder_seq_length):
            decoder_input_data = pad_sequences([curr_seq], maxlen=self.max_decoder_seq_length, padding='post')

            # Prédiction du mot suivant
            predictions = model.predict([encoder_input_data, decoder_input_data], verbose=False)
            next_word_idx = np.argmax(predictions[0], axis=-1)

            if next_word_idx[i] == 0:  # Si c'est un padding, on continue
                continue

            curr_seq.append(next_word_idx[i])
            next_word = self.tokenizer.index_word[next_word_idx[i]]
            generated_text.append(next_word)

            # Si le token '</sonnet>' est atteint, arrêter la prédiction
            if next_word == '</sonnet>':
                break

        return ' '.join(generated_text)