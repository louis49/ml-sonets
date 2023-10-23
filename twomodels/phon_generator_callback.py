import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.python.keras.callbacks import Callback
from data import Data

class PhonGenerator(Callback):

    def __init__(self, model, data: Data, every_epoch=1):
        super().__init__()
        self.data = data
        self.every_epoch = every_epoch
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0:
            text_1 = "SONNET POUR LA MORT DE SON AMIE"
            text_2 = "Amour fou"
            generated_text_1 = self.generate_text(text_1)
            generated_text_2 = self.generate_text(text_2)
            print("\nSample generation at epoch {}: \n{} \n{}".format(epoch + 1, text_1 + ' ' + generated_text_1, text_2 + ' ' + generated_text_2))

    def generate_text(self, input_title):
        tokenized_title = self.data.title_tokenizer.texts_to_sequences([input_title])[0]
        padded_title = pad_sequences([tokenized_title], maxlen=self.data.title_max_size, padding='post')

        seed_text = '$'
        output_sequence = self.data.phon_tokenizer.texts_to_sequences([seed_text])[0]

        for i in range(self.data.phon_max_size * 14 + 1):
            # Préparation de l'input pour le modèle, ajustement à la longueur correcte pour chaque nouvelle prédiction
            current_sequence = pad_sequences([output_sequence], maxlen=self.data.phon_max_size * 14 + 1, padding='post')

            input_data = [np.array(padded_title), np.array(current_sequence)]

            # Prédiction du prochain mot
            predictions = self.model.predict(input_data, verbose=False)

            # Extraction de l'index du mot (ce pourrait être la valeur maximale dans les prédictions, i.e., np.argmax)
            next_word_idx = np.argmax(predictions[0], axis=-1)

            # Ajout du mot prédit à la séquence de sortie
            output_sequence.append(next_word_idx[i])

            next_word = self.data.phon_tokenizer.index_word[next_word_idx[i]]

            # Ajout du mot prédit au verset
            seed_text += next_word
        text = seed_text[1:-1]
        divided_strings = [text[i:i + 3] for i in range(0, len(text), 3)]
        divided_strings_clean = [string.replace('.', '') for string in divided_strings]
        return self.data.convertir_rimes_en_lettres(divided_strings_clean) + " " + text.replace('.', ' ')