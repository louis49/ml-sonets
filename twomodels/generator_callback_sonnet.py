import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.python.keras.callbacks import Callback
from data import Data
import tensorflow as tf

class SonnetGenerator(Callback):

    def __init__(self, model, data: Data, every_epoch=1):
        super().__init__()
        self.data = data
        self.every_epoch = every_epoch
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0:
            text_1 = "SONNET POUR LA MORT DE SON AMIE"
            text_2 = "Amour fou"
            text_3 = ""
            generated_text_1 = self.generate_text(text_1)
            generated_text_2 = self.generate_text(text_2)
            generated_text_3 = self.generate_text(text_3)
            print("\nSample generation at epoch {}: \n{} \n{} \n{}".format(epoch + 1, text_1 + ' ' + generated_text_1, text_2 + ' ' + generated_text_2, 'VIDE ' + generated_text_3))

    def generate_text(self, input_title):
        tokenized_title = self.data.title_tokenizer.texts_to_sequences([input_title])[0]
        padded_title = pad_sequences([tokenized_title], maxlen=self.data.title_max_size, padding='post')

        output_text_sequence = self.data.text_tokenizer.texts_to_sequences(['<start>'])[0]
        output_phon_sequence = self.data.phon_tokenizer.texts_to_sequences(['$'])[0]

        seed_text = ""

        for i in range(14):
            for x in range(self.data.phon_max_size):

                current_text_sequence = pad_sequences([output_text_sequence], maxlen=self.data.text_max_size * 14 - 14,
                                                      padding='post')
                current_phon_sequence = pad_sequences([output_phon_sequence], maxlen=self.data.phon_max_size * 14 + 1, padding='post')[0]

                current_phon_one_hot = tf.keras.utils.to_categorical(current_phon_sequence, num_classes=self.data.phon_words + 1)

                input_data = [np.array(padded_title), current_phon_one_hot[np.newaxis, ...], np.array(current_text_sequence)]

                predictions = self.model.predict(input_data, verbose=False)

                phons_prediction = predictions[0]

                next_phon_idx = np.argmax(phons_prediction, axis=-1)

                output_phon_sequence.append(next_phon_idx[i*x])

            for y in range(self.data.text_max_size):
                current_text_sequence = pad_sequences([output_text_sequence], maxlen=self.data.text_max_size * 14 - 14,
                                                      padding='post')
                current_phon_sequence = \
                pad_sequences([output_phon_sequence], maxlen=self.data.phon_max_size * 14 + 1, padding='post')[0]

                current_phon_one_hot = tf.keras.utils.to_categorical(current_phon_sequence,
                                                                     num_classes=self.data.phon_words + 1)

                input_data = [np.array(padded_title), current_phon_one_hot[np.newaxis, ...],
                              np.array(current_text_sequence)]

                predictions = self.model.predict(input_data, verbose=False)

                text_prediction = predictions[1]

                next_word_idx = np.argmax(text_prediction, axis=-1)

                output_text_sequence.append(next_word_idx[i * y])

                next_word = self.data.text_tokenizer.index_word[next_word_idx[i * y]]

                seed_text += next_word
        text = seed_text[1:-1]
        divided_strings = [text[i:i + 3] for i in range(0, len(text), 3)]
        reversed_divided_strings = [string[::-1] for string in divided_strings]
        divided_strings_clean = [string.replace('.', '') for string in reversed_divided_strings]
        return self.data.convertir_rimes_en_lettres(divided_strings_clean) + " " + text.replace('.', ' ')