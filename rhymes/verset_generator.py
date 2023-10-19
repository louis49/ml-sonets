import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.python.keras.callbacks import Callback


class VersetGenerator(Callback):

    def __init__(self, title, phon, title_tokenizer, text_tokenizer, phon_tokenizer, title_words, max_len_text, max_len_phon, max_len_title, every_epoch=1):
        super(VersetGenerator, self).__init__()
        self.title = title
        self.phon = phon
        self.title_tokenizer = title_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phon_tokenizer = phon_tokenizer
        self.title_words = title_words
        self.max_len_text = max_len_text
        self.max_len_title = max_len_title
        self.max_len_phon = max_len_phon
        self.every_epoch = every_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_epoch == 0:
            generated_text = self.generate_text(self.model)
            print("\nSample generation at epoch {}: \n{}".format(epoch + 1, generated_text))

    def generate_text(self, model):
        # Préparation des entrées initiales
        input_title = self.title
        input_phon = self.phon

        # Tokenisation des entrées
        tokenized_title = self.title_tokenizer.texts_to_sequences([input_title])[0]
        tokenized_phon = self.phon_tokenizer.texts_to_sequences([input_phon])[0]

        # Ajustement des séquences à la longueur maximale en ajoutant du padding si nécessaire
        tokenized_title = pad_sequences([tokenized_title], maxlen=self.max_len_title, padding='post')
        tokenized_phon = pad_sequences([tokenized_phon], maxlen=self.max_len_phon, padding='post')

        # Initialisation de la séquence de sortie/seed avec un mot (ou vide)
        seed_text = '<start>'
        output_sequence = [self.text_tokenizer.texts_to_sequences([seed_text])[0][
                               0]]  # Prenez le premier élément de la séquence tokenisée

        for i in range(self.max_len_text):
            # Préparation de l'input pour le modèle, ajustement à la longueur correcte pour chaque nouvelle prédiction
            current_sequence = pad_sequences([output_sequence], maxlen=self.max_len_text - 1, padding='post')

            input_data = [np.array(tokenized_title), np.array(tokenized_phon), np.array(current_sequence)]

            # Prédiction du prochain mot
            predictions = model.predict(input_data, verbose=False)

            # Extraction de l'index du mot (ce pourrait être la valeur maximale dans les prédictions, i.e., np.argmax)
            next_word_idx = np.argmax(predictions[0], axis=-1)

            # Si le modèle retourne un index 0, cela signifie souvent qu'il est indécis, nous devrions alors arrêter la prédiction ici
            if next_word_idx[i] == 0:
                break

            # Ajout du mot prédit à la séquence de sortie
            output_sequence.append(next_word_idx[i])

            next_word = self.text_tokenizer.index_word[next_word_idx[i]]

            # Si le mot prédit est '<end>', nous arrêtons la génération de texte
            if next_word == '<end>':
                break

            # Ajout du mot prédit au verset
            seed_text += ' ' + next_word

        return seed_text.strip().replace('<start>', '')  # retourne le texte généré, sans espace superflu au début ou à la fin
